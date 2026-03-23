# Contributing to AA-YOLO

Thank you for your interest in contributing to AA-YOLO! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AMIAD-Research/AA-YOLO.git
cd AA-YOLO

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install ruff black pytest pre-commit

# Install pre-commit hooks
pre-commit install
```

## Code Style

- **Formatter**: Black (line-length: 120)
- **Linter**: Ruff
- **Type hints**: Required for new public APIs
- **Docstrings**: Google style, required for new classes and public methods

Run before committing:
```bash
ruff check .
black --check .
pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, atomic commits
3. Add/update tests for new functionality
4. Update documentation if needed
5. Ensure CI passes (lint + tests)
6. Submit PR with a clear description

## Project Structure

```
AA-YOLO/
├── cfg/training/          # Model architecture configs (YAML)
├── data/                  # Dataset configs + hyperparameters
│   ├── datasets/          # Dataset images and labels
│   ├── hyp.scratch.AA_yolo.yaml  # Training hyperparameters
│   └── *.yaml             # Dataset definitions
├── models/                # Model definitions
│   ├── common.py          # Building blocks (Conv, filtering2D, anomaly_testing...)
│   ├── yolo.py            # Detection heads (IDetect_AA) and Model class
│   └── experimental.py    # Experimental modules
├── utils/                 # Training utilities
│   ├── loss.py            # Loss functions (ComputeLoss, ComputeLossOTA)
│   ├── datasets.py        # Data loading and augmentation
│   ├── general.py         # General utilities
│   └── metrics.py         # Evaluation metrics
├── tests/                 # Unit and smoke tests
├── train.py               # Training script
├── test.py                # Evaluation script
├── detect.py              # Inference script
└── ARCHITECTURE.md        # Technical architecture documentation
```

## Reporting Issues

When opening an issue, please include:
- Python and PyTorch versions
- GPU model and CUDA version (if applicable)
- Full error traceback
- Steps to reproduce
