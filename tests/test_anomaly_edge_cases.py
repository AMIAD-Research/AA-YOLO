"""Edge-case tests for AA-YOLO anomaly modules.

Tests numerical stability, gradient flow, and EMA convergence
under extreme conditions that occur in real IR scenes.
"""

import pytest
import torch


class TestAnomalyTesting:
    """Edge-case tests for anomaly_testing module."""

    def _make_module(self, **kwargs):
        from models.common import anomaly_testing
        defaults = dict(alpha=0.05, ema_momentum=0.1)
        defaults.update(kwargs)
        return anomaly_testing(**defaults)

    def test_all_zero_input(self):
        """All-zero features should not produce NaN (common in masked IR regions)."""
        module = self._make_module()
        module.train()
        x = torch.zeros(2, 8, 10, 10)
        out = module(x)
        assert not torch.isnan(out).any(), "NaN with zero input"
        assert not torch.isinf(out).any(), "Inf with zero input"

    def test_extreme_large_values(self):
        """Extreme feature values (>1e3) should not cause overflow."""
        module = self._make_module()
        module.train()
        x = torch.randn(2, 8, 10, 10) * 1e4
        out = module(x)
        assert not torch.isnan(out).any(), "NaN with extreme input"
        assert not torch.isinf(out).any(), "Inf with extreme input"

    def test_very_small_positive_values(self):
        """Very small positive values (thermal low-intensity scenes)."""
        module = self._make_module()
        module.train()
        x = torch.rand(2, 8, 10, 10) * 1e-6
        out = module(x)
        assert not torch.isnan(out).any(), "NaN with tiny input"
        assert not torch.isinf(out).any(), "Inf with tiny input"

    def test_ema_convergence(self):
        """EMA buffer should converge over repeated forward passes."""
        module = self._make_module(ema_momentum=0.3)
        module.train()
        target_mean = 5.0
        for _ in range(50):
            x = torch.ones(2, 4, 8, 8) * target_mean + torch.randn(2, 4, 8, 8) * 0.1
            module(x)
        # EMA should be close to target_mean
        assert abs(module.lambda_ema.mean().item() - target_mean) < 1.0, \
            f"EMA did not converge: {module.lambda_ema.mean().item():.2f} vs {target_mean}"

    def test_train_vs_eval_consistency(self):
        """Output shape and validity should be consistent across modes."""
        module = self._make_module()
        x = torch.randn(2, 8, 10, 10)
        # Train mode
        module.train()
        out_train = module(x.clone())
        # Eval mode
        module.eval()
        out_eval = module(x.clone())
        assert out_train.shape == out_eval.shape
        assert not torch.isnan(out_eval).any()

    def test_output_range(self):
        """Anomaly scores should be in [-1, 1] range (2*sigmoid - 1)."""
        module = self._make_module()
        module.train()
        x = torch.randn(2, 8, 10, 10) * 10
        out = module(x)
        assert out.min() >= -1.0 - 1e-6, f"Output below -1: {out.min()}"
        assert out.max() <= 1.0 + 1e-6, f"Output above 1: {out.max()}"


class TestLnGamma:
    """Edge-case tests for lnGamma autograd function."""

    def test_very_small_x(self):
        """Very small x values near zero."""
        from models.common import lnGamma
        x = torch.tensor([1e-8, 1e-6, 1e-4], requires_grad=True)
        a = torch.tensor(3)
        out = lnGamma.apply(x, a)
        assert not torch.isnan(out).any()
        out.sum().backward()
        assert not torch.isnan(x.grad).any()

    def test_large_x(self):
        """Large x values should not overflow."""
        from models.common import lnGamma
        x = torch.tensor([100.0, 500.0, 1000.0], requires_grad=True)
        a = torch.tensor(3)
        out = lnGamma.apply(x, a)
        assert not torch.isnan(out).any()
        out.sum().backward()
        assert not torch.isnan(x.grad).any()

    def test_gradient_clamping(self):
        """Gradients should be clamped and never exceed 1e4."""
        from models.common import lnGamma
        x = torch.tensor([1e-10, 1e-8], requires_grad=True)
        a = torch.tensor(1)
        out = lnGamma.apply(x, a)
        out.sum().backward()
        assert (x.grad.abs() <= 1e4 + 1e-3).all(), f"Gradient too large: {x.grad}"


class TestFiltering2D:
    """Edge-case tests for filtering2D module."""

    def test_single_channel(self):
        """Single output channel."""
        from models.common import filtering2D
        module = filtering2D(16, 1)
        x = torch.randn(1, 16, 8, 8)
        out = module(x)
        assert out.shape == (1, 1, 8, 8)

    def test_gradient_flows(self):
        """Gradients should flow through the full filtering pipeline."""
        from models.common import filtering2D
        module = filtering2D(8, 4)
        x = torch.randn(2, 8, 10, 10, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
