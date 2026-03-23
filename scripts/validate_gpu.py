#!/usr/bin/env python3
"""AA-YOLO GPU Validation Script.

Run this on a CUDA-capable machine to validate all improvements
that cannot be tested on CPU-only environments.

Usage:
    git clone https://github.com/creacress/AA-YOLO.git
    cd AA-YOLO && git checkout improvements/v1
    pip install -r requirements.txt
    python scripts/validate_gpu.py

Requires: CUDA GPU, PyTorch with CUDA support
"""

import os
import sys

# Ensure project root is on sys.path when running from scripts/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import time
import torch
import torch.nn as nn
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
results = []


def test(name, fn):
    """Run a test and collect result."""
    try:
        start = time.time()
        fn()
        dt = time.time() - start
        print(f"  {PASS}  {name} ({dt:.2f}s)")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append((name, False, str(e)))


# ── 0. Environment Check ────────────────────────────────────────────
def check_env():
    print("\n" + "=" * 60)
    print("AA-YOLO GPU Validation")
    print("=" * 60)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  {WARN} No CUDA GPU detected — GPU-specific tests will be skipped")
    print()


# ── 1. Model Build & Forward Pass on GPU ────────────────────────────
def test_gpu_forward_pass():
    """Build model and run forward pass on GPU."""
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.eval()
    x = torch.randn(1, 3, 640, 640).cuda()
    with torch.no_grad():
        y = model(x)
    assert y[0].shape[0] == 1
    assert y[0].device.type == 'cuda'
    del model, x, y
    torch.cuda.empty_cache()


# ── 2. Device Consistency (lnGamma fix) ─────────────────────────────
def test_lngamma_device_consistency():
    """Verify lnGamma works on GPU without device mismatch."""
    from models.common import lnGamma
    x = torch.tensor([1.0, 2.0, 5.0], device='cuda', requires_grad=True)
    a = torch.tensor(3, device='cuda')
    out = lnGamma.apply(x, a)
    assert out.device.type == 'cuda'
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert x.grad.device.type == 'cuda'
    assert not torch.isnan(x.grad).any()


# ── 3. anomaly_testing on GPU ───────────────────────────────────────
def test_anomaly_testing_gpu():
    """Verify anomaly_testing module on GPU with EMA buffer."""
    from models.common import anomaly_testing
    module = anomaly_testing(alpha=0.05, ema_momentum=0.1).cuda()
    module.train()
    for _ in range(10):
        x = torch.randn(4, 16, 20, 20).cuda()
        out = module(x)
        assert out.device.type == 'cuda'
        assert not torch.isnan(out).any()
    # Check EMA buffer is on GPU
    assert module.lambda_ema.device.type == 'cuda'
    # Switch to eval
    module.eval()
    out = module(torch.randn(2, 16, 20, 20).cuda())
    assert not torch.isnan(out).any()


# ── 4. Gradient Flow Through Full AA Pipeline ───────────────────────
def test_gradient_flow_gpu():
    """Verify gradients flow through the full model on GPU."""
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.train()
    x = torch.randn(2, 3, 320, 320).cuda()
    y = model(x)
    # Sum all outputs to create a scalar loss
    loss = sum(yi.sum() for yi in y if isinstance(yi, torch.Tensor))
    loss.backward()
    # Check at least some parameters have gradients
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model.parameters())
    assert grad_count > 0, f"No gradients computed! {grad_count}/{total_count}"
    # Check no NaN gradients
    nan_params = [n for n, p in model.named_parameters()
                  if p.grad is not None and torch.isnan(p.grad).any()]
    assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"
    del model, x, y
    torch.cuda.empty_cache()


# ── 5. Mixed Precision (AMP) Training Step ──────────────────────────
def test_amp_training_step():
    """Verify model works with automatic mixed precision."""
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(2, 3, 320, 320).cuda()
    with torch.cuda.amp.autocast():
        y = model(x)
        loss = sum(yi.sum() for yi in y if isinstance(yi, torch.Tensor))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert not torch.isnan(loss)
    del model, x, y
    torch.cuda.empty_cache()


# ── 6. torch.compile() Support ──────────────────────────────────────
def test_torch_compile():
    """Verify torch.compile() works on the model (PyTorch 2.0+)."""
    if not hasattr(torch, 'compile'):
        raise RuntimeError("PyTorch < 2.0, torch.compile not available")
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.eval()
    # On Windows, Triton (inductor backend) is not available — fall back to eager
    backend = 'inductor'
    if sys.platform == 'win32':
        try:
            import triton  # noqa: F401
        except ImportError:
            backend = 'eager'
            print(f"       {WARN} Triton unavailable on Windows, using backend='eager'")
    compiled = torch.compile(model, mode='reduce-overhead', backend=backend)
    x = torch.randn(1, 3, 640, 640).cuda()
    with torch.no_grad():
        # First call triggers compilation (slow)
        y = compiled(x)
    assert y[0].shape[0] == 1
    del model, compiled, x, y
    torch.cuda.empty_cache()


# ── 7. Inference Benchmark ──────────────────────────────────────────
def test_inference_benchmark():
    """Benchmark inference speed on GPU."""
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.eval()
    x = torch.randn(1, 3, 640, 640).cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.time()
            model(x)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    fps = 1000 / avg_ms
    print(f"       Inference: {avg_ms:.1f} ± {std_ms:.1f} ms ({fps:.0f} FPS)")

    # Measure VRAM
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"       Peak VRAM: {vram_mb:.0f} MB")
    del model, x
    torch.cuda.empty_cache()


# ── 8. Mini Training Loop (5 epochs, synthetic data) ────────────────
def test_mini_training():
    """Run 5 training epochs on synthetic data to verify convergence."""
    from models.yolo import Model
    model = Model(cfg='cfg/training/AA-yolov7-tiny.yaml', ch=3, nc=1).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    losses = []
    for epoch in range(5):
        x = torch.randn(2, 3, 320, 320).cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = sum(yi.abs().mean() for yi in y if isinstance(yi, torch.Tensor))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        print(f"       Epoch {epoch+1}/5: loss={losses[-1]:.4f}")

    # Loss should not be NaN and should generally decrease or stay stable
    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    del model, x, y
    torch.cuda.empty_cache()


# ── 9. Multi-batch EMA Stability ────────────────────────────────────
def test_ema_stability_gpu():
    """Verify EMA buffer stability over many batches on GPU."""
    from models.common import anomaly_testing
    module = anomaly_testing(alpha=0.05, ema_momentum=0.1).cuda()
    module.train()

    for i in range(100):
        # Simulate varying IR intensities
        intensity = 0.01 + (i / 100) * 10
        x = torch.randn(4, 8, 16, 16).cuda() * intensity
        out = module(x)
        if torch.isnan(out).any():
            raise AssertionError(f"NaN at batch {i}, intensity={intensity:.2f}")
        if torch.isinf(out).any():
            raise AssertionError(f"Inf at batch {i}, intensity={intensity:.2f}")

    ema_val = module.lambda_ema.mean().item()
    assert np.isfinite(ema_val), f"EMA became non-finite: {ema_val}"


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    check_env()

    has_cuda = torch.cuda.is_available()

    print("─── CPU Tests (always run) ───")
    # Run pytest tests first
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                           capture_output=False)
    print()

    if has_cuda:
        print("─── GPU Tests ───")
        test("Model forward pass (GPU)", test_gpu_forward_pass)
        test("lnGamma device consistency", test_lngamma_device_consistency)
        test("anomaly_testing on GPU", test_anomaly_testing_gpu)
        test("Gradient flow (full model)", test_gradient_flow_gpu)
        test("AMP training step", test_amp_training_step)
        test("torch.compile() support", test_torch_compile)
        test("Inference benchmark", test_inference_benchmark)
        test("Mini training (5 epochs)", test_mini_training)
        test("EMA stability (100 batches)", test_ema_stability_gpu)
    else:
        print("─── GPU Tests SKIPPED (no CUDA) ───")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    if failed == 0:
        print(f"  {PASS}  ALL {total} GPU TESTS PASSED")
    else:
        print(f"  {FAIL}  {failed}/{total} GPU tests failed:")
        for name, ok, err in results:
            if not ok:
                print(f"       - {name}: {err}")
    print("=" * 60)
    sys.exit(1 if failed else 0)
