"""Smoke tests for AA-YOLO model loading and forward pass."""

import pytest
import torch
import yaml


def test_config_loads():
    """Test that model config YAML can be loaded."""
    with open("cfg/training/AA-yolov7-tiny.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "backbone" in cfg
    assert "head" in cfg
    assert cfg["nc"] == 80


def test_hyperparams_load():
    """Test that hyperparameter file loads with AA-YOLO specific params."""
    with open("data/hyp.scratch.AA_yolo.yaml") as f:
        hyp = yaml.safe_load(f)
    assert "loss_AA" in hyp
    assert "anomaly_alpha" in hyp
    assert "anomaly_ema_momentum" in hyp
    assert hyp["anomaly_alpha"] == 0.05


def test_anomaly_testing_module():
    """Test anomaly_testing module forward pass."""
    from models.common import anomaly_testing

    module = anomaly_testing(alpha=0.05, ema_momentum=0.1)
    module.train()
    x = torch.randn(2, 8, 10, 10)
    out = module(x)
    assert out.shape == (2, 1, 10, 10)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_filtering2d_module():
    """Test filtering2D module forward pass."""
    from models.common import filtering2D

    module = filtering2D(64, 24)
    x = torch.randn(2, 64, 20, 20)
    out = module(x)
    assert out.shape == (2, 24, 20, 20)


def test_ln_gamma_forward():
    """Test lnGamma custom autograd function."""
    from models.common import lnGamma

    x = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
    a = torch.tensor(3)
    out = lnGamma.apply(x, a)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # Test backward
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
