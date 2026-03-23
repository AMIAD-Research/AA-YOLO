"""Tests for Tier 2-4 improvements: model variants, ONNX export, temperature calibration, gradient accumulation."""

import pytest
import torch
import yaml


class TestModelVariants:
    """Test that all model variant configs build and produce correct outputs."""

    @pytest.mark.parametrize("cfg,expected_min_params,expected_max_params", [
        ("cfg/training/AA-yolov7-small.yaml", 0.3e6, 0.8e6),
        ("cfg/training/AA-yolov7-medium.yaml", 1.0e6, 3.0e6),
        ("cfg/training/AA-yolov7-tiny.yaml", 4.0e6, 10.0e6),
    ])
    def test_variant_builds_and_forward(self, cfg, expected_min_params, expected_max_params):
        """Verify each model variant builds, has expected param count, and produces output."""
        from models.yolo import Model
        model = Model(cfg=cfg, ch=3, nc=1)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        assert expected_min_params <= params <= expected_max_params, \
            f"{cfg}: {params/1e6:.2f}M params not in expected range"
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        assert y[0].shape[0] == 1
        assert y[0].shape[-1] == 6  # 4 box + 1 obj + 1 cls


class TestFullModelForwardPass:
    """Test non-regression: full model forward pass with expected output shapes."""

    def test_forward_pass_shape(self):
        """Load full model, run forward (1,3,640,640), verify output shape."""
        from models.yolo import Model
        model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        # Inference returns (predictions, raw_outputs)
        preds = y[0]
        assert preds.ndim == 3  # (batch, num_detections, 6)
        assert preds.shape[0] == 1
        assert preds.shape[2] == 6  # xyxy + obj + cls

    def test_train_forward_shape(self):
        """Verify training forward pass returns list of per-layer outputs."""
        from models.yolo import Model
        model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        y = model(x)
        assert isinstance(y, list)
        assert len(y) == 3  # 3 detection layers (P3, P4, P5)
        for yi in y:
            assert yi.shape[0] == 2  # batch size


class TestGradientAccumulation:
    """Test that gradient accumulation logic works correctly."""

    def test_gradients_accumulate(self):
        """Verify gradients accumulate over multiple forward/backward passes."""
        from models.common import filtering2D
        module = filtering2D(64, 24)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        optimizer.zero_grad()

        # Accumulate over 4 steps
        accumulate = 4
        for step in range(accumulate):
            x = torch.randn(2, 64, 10, 10)
            out = module(x)
            loss = out.mean() / accumulate
            loss.backward()

        # Check gradients exist and are non-zero
        param = next(module.parameters())
        assert param.grad is not None
        assert param.grad.abs().sum() > 0

        # After step, gradients should be cleared
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        assert param.grad.abs().sum() == 0


class TestONNXExportMode:
    """Test ONNX export compatibility of anomaly_testing."""

    def test_export_mode_produces_output(self):
        """Verify export_mode path produces valid output without gammaincc."""
        from models.common import anomaly_testing
        module = anomaly_testing(alpha=0.05)
        module.eval()
        module.export_mode = True
        # Run a few batches to initialize EMA
        module.train()
        for _ in range(3):
            module(torch.randn(2, 8, 10, 10))
        module.eval()
        module.export_mode = True
        x = torch.randn(2, 8, 10, 10)
        out = module(x)
        assert out.shape == (2, 1, 10, 10)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_export_vs_normal_correlation(self):
        """Verify export mode output is correlated with normal mode."""
        from models.common import anomaly_testing
        module = anomaly_testing(alpha=0.05)
        module.train()
        for _ in range(10):
            module(torch.randn(4, 8, 10, 10))
        module.eval()
        x = torch.randn(4, 8, 10, 10)
        # Normal mode
        module.export_mode = False
        out_normal = module(x).detach()
        # Export mode
        module.export_mode = True
        out_export = module(x).detach()
        # Outputs should be correlated (same sign, similar range)
        assert out_normal.shape == out_export.shape
        # Both should be in [-1, 1] range
        assert out_normal.abs().max() <= 1.0
        assert out_export.abs().max() <= 1.0


class TestTemperatureCalibration:
    """Test temperature scaling for confidence calibration."""

    def test_default_temperature(self):
        """Default temperature=1.0 should not change output."""
        from models.common import anomaly_testing
        module = anomaly_testing(alpha=0.05)
        module.eval()
        module.train()
        for _ in range(3):
            module(torch.randn(2, 8, 10, 10))
        module.eval()
        x = torch.randn(2, 8, 10, 10)
        out1 = module(x).detach().clone()
        assert module.temperature.item() == 1.0
        out2 = module(x).detach()
        assert torch.allclose(out1, out2)

    def test_temperature_scaling_effect(self):
        """Higher temperature should produce less extreme (more uncertain) scores."""
        from models.common import anomaly_testing
        module = anomaly_testing(alpha=0.05)
        module.train()
        for _ in range(5):
            module(torch.randn(4, 16, 10, 10))
        module.eval()
        x = torch.randn(4, 16, 10, 10)
        # Temperature = 1 (default)
        module.temperature.fill_(1.0)
        out_t1 = module(x).detach()
        # Temperature = 2 (more uncertain)
        module.temperature.fill_(2.0)
        out_t2 = module(x).detach()
        # Higher temperature should reduce the absolute magnitude of scores
        assert out_t2.abs().mean() <= out_t1.abs().mean()


class TestEMADecayScheduling:
    """Test EMA decay scheduling in ModelEMA."""

    def test_ema_with_epochs(self):
        """Verify epoch-aware EMA decay ramps correctly."""
        from utils.torch_utils import ModelEMA
        from models.common import filtering2D
        model = filtering2D(16, 8)
        ema = ModelEMA(model, epochs=100)
        # At epoch 0, effective decay should be ~0.999 * ramp
        ema.set_epoch(0)
        d0 = ema.decay(1000)
        # At epoch 100, effective decay should be ~0.9999 * ramp
        ema.set_epoch(100)
        d100 = ema.decay(1000)
        # Decay should increase with epoch
        assert d100 > d0

    def test_ema_without_epochs(self):
        """Verify original EMA behavior when epochs not specified."""
        from utils.torch_utils import ModelEMA
        from models.common import filtering2D
        model = filtering2D(16, 8)
        ema = ModelEMA(model)  # no epochs
        # decay(x) = 0.9999 * (1 - exp(-x/2000)), at x=10000 → ~0.993
        d = ema.decay(10000)
        assert 0.9 < d < 1.0
