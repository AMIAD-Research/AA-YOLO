# AA-YOLO Architecture

## Overview

AA-YOLO (Anomaly-Aware YOLO) extends YOLOv7 with a novel detection head that replaces the standard learned objectness prediction with a statistical anomaly testing approach. This makes the detector more robust under limited training data, sensor noise, and domain shifts — common challenges in Infrared Small Target Detection (IRSTD).

## Architecture Diagram

```
Input Image (640x640)
       │
       ▼
┌─────────────────────┐
│   YOLOv7-tiny       │
│   Backbone           │
│   (E-ELAN blocks)    │
│                      │
│   P3 ──► 128ch, /8  │
│   P4 ──► 256ch, /16 │
│   P5 ──► 512ch, /32 │
└───┬───┬───┬─────────┘
    │   │   │
    ▼   ▼   ▼
┌─────────────────────┐
│   FPN + PAN Neck     │
│   (top-down +        │
│    bottom-up fusion) │
└───┬───┬───┬─────────┘
    │   │   │
    ▼   ▼   ▼
┌─────────────────────────────────────┐
│       IDetect_AA Head               │
│                                     │
│  For each scale (P3, P4, P5):       │
│                                     │
│  features ──┬──► Conv 1x1 ──► box   │
│             │    + ImplicitA/M  cls  │
│             │                        │
│             └──► filtering2D ──►     │
│                  anomaly_testing ──► │
│                  objectness score    │
│                                     │
│  Output: [x, y, w, h, obj, cls]     │
└─────────────────────────────────────┘
```

## Key Components

### IDetect_AA (models/yolo.py)

The anomaly-aware detection head. Unlike standard YOLO heads where objectness is predicted by a learned convolution, IDetect_AA computes objectness through statistical testing:

- **Box regression + classification**: Standard 1x1 convolution with ImplicitA (additive) and ImplicitM (multiplicative) knowledge modules
- **Objectness**: Computed via `filtering2D` → `anomaly_testing` pipeline

### filtering2D (models/common.py)

A lightweight 2-layer CNN that transforms backbone features before anomaly testing:

```
Input (ch1) → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → Output (ch2)
```

The output channels are `na × aa_chan` (number of anchors × anomaly channels per anchor), which are then reshaped for per-anchor anomaly testing.

### anomaly_testing (models/common.py)

The core innovation. Computes anomaly scores using statistical hypothesis testing:

1. **Channel normalization**: Compute per-sample mean, normalize features using an adaptive rate (EMA-smoothed during training)
2. **L1 norm**: Compute L1 norm across channels, producing a scalar per spatial location
3. **Log upper incomplete gamma**: Apply `lnGamma` — the log of the upper incomplete gamma function. Under the null hypothesis (no target), normalized features follow an exponential distribution, and the L1 norm follows a Gamma distribution
4. **Sigmoid activation**: Scale and shift to [0, 1] range via `2σ(α·x) - 1`

**Hyperparameters:**
- `alpha` (default: 0.05): Controls sigmoid steepness. Higher = sharper discrimination
- `ema_momentum` (default: 0.1): EMA update rate for channel statistics
- `inference_ema_weight` (default: 0.15): Weight of historical EMA vs batch statistics at inference

### lnGamma (models/common.py)

Custom autograd function computing `log(Γ_upper(a, x))` with numerically stable gradients. Falls back to asymptotic expansion when standard computation yields inf/nan.

## Loss Functions

### ComputeLoss / ComputeLossOTA (utils/loss.py)

When `loss_AA=1` in hyperparameters:
- Objectness loss switches from `BCEWithLogitsLoss` to `MSELoss`
- This is because anomaly scores are continuous values in [0, 1], not logits

The total loss is: `L = λ_box · L_CIoU + λ_obj · L_obj + λ_cls · L_cls`

## Configuration

### Model config: `cfg/training/AA-yolov7-tiny.yaml`

The detection head is defined at the last line:
```yaml
[[74,75,76], 1, IDetect_AA, [nc, anchors, 8]]
```
Where `8` is `aa_chan` (anomaly channels per anchor).

### Hyperparameters: `data/hyp.scratch.AA_yolo.yaml`

Key AA-YOLO specific parameters:
- `loss_AA: 1` — Enable anomaly-aware loss (MSELoss for objectness)
- `anomaly_alpha: 0.05` — Sigmoid scaling factor
- `anomaly_ema_momentum: 0.1` — EMA momentum
- `anomaly_inference_ema_weight: 0.15` — Inference EMA blend weight

## Datasets

| Dataset | Images | Targets | Resolution | Link |
|---------|--------|---------|------------|------|
| SIRST | 427 | 480 | Mixed | [GitHub](https://github.com/YimianDai/sirst) |
| IRSTD-1k | 1000 | 1495 | 512×512 | [GitHub](https://github.com/RuiZhang97/ISNet) |

Labels are provided in YOLO format in `data/datasets/*/labels/`.
