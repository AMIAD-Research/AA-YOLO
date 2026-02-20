# 🔍 AA-YOLO: Official Implementation of "An Anomaly-Aware Detection Head for Frugal and Robust Infrared Small Target Detection"

Welcome to the official implementation repository of our [paper](https://doi.org/10.1016/j.engappai.2026.114186)! This project is a Python implementation of the Anomaly-Aware version of YOLO detector, as described in our work. 📝

## 🚀 Features

- **Improved Performance:** AA-YOLO achieves competitive performance on IRSTD benchmarks. 🏆
- **Robustness:** Demonstrates robustness in limited data, noise, and domain shifts scenarios. 🔒
- **Versatility:** Applicable across various YOLO backbones, including lightweight models and instance segmentation YOLO. 🌐

## 📥 Getting Started

### Installation

1. Clone this repository: `git clone https://github.com/AMIAD-Research/AA-YOLO.git`
2. Follow the [yolov7 installation instructions](https://github.com/WongKinYiu/yolov7) to install dependencies and build the project. 🔧

You can also install the depedencies by running the following command: 
```uv sync```
3. Prepare the datasets in the YOLO format (see below for more details). 📁

### Dataset Preparation

For the SIRST dataset, dowload the [dataset](https://github.com/YimianDai/sirst) and place the images in `data/datasets/SIRST/images`. For the IRSTD-1k dataset, dowload the [dataset](https://github.com/RuiZhang97/ISNet) and place the images in `data/datasets/IRSTD-1k/images`. 

The YOLO format labels are already provided in this repo. 

For custom datasets, follow the same architecture and place the dataset in a new directory under `data/datasets`.


## 🔬 Testing

We provide our best weights trained on IRSTD-1k and SIRST datasets in the `best_model_AA_YOLOv7t` folder. To test a model, run:

```python /path/to/test.py
# test AA-YOLOv7t model on IRSTD-1K dataset
python test.py --batch-size 16 --exist-ok --data data/irstd_1k_eflnet.yaml --img-size 640 --iou-thres 0.05 --task test --weights best_model_AA_YOLOv7t/irstd1k_best.pt --name test_AAyolov7t_irstd --single-cls

# test AA-YOLOv7t model on SIRST dataset 
python test.py --batch-size 16 --exist-ok --data data/sirst.yaml --img-size 640 --iou-thres 0.05 --task test --weights best_model_AA_YOLOv7t/sirst_best.pt --name test_AAyolov7t_sirst --single-cls
```
The results are saved in the `runs/test` folder. Precision, recall, F1 score and Average Precision (AP) metrics can be found in the `results.txt` file.


## 📚 Training

To train a model, use the following commands:

```python /path/to/train.py
# train AA-YOLOv7t model on IRSTD-1K dataset
python train.py --workers 8 --batch-size 16 --data data/irstd_1k_eflnet.yaml --img 640 640 --iou-thres 0.05 --epochs 600 --cfg cfg/training/AA-yolov7-tiny.yaml --name test_irstd_AA_yolov7t --hyp data/hyp.scratch.AA_yolo.yaml --single-cls

# train AA-YOLOv7t model on SIRST dataset 
python train.py --workers 8 --batch-size 16 --data data/sirst.yaml --img 640 640 --iou-thres 0.05 --epochs 600 --cfg cfg/training/AA-yolov7-tiny.yaml --name test_sirst_AA_yolov7t --hyp data/hyp.scratch.AA_yolo.yaml --single-cls
```

## 📝 Citing this Work

Please cite using the bibtex generated from [the EAAI article page](https://doi.org/10.1016/j.engappai.2026.114186)

for example as:

```
@article{CIOCARLAN2026114186,
title = {An anomaly-aware detection head for frugal and robust Infrared Small Target Detection},
journal = {Engineering Applications of Artificial Intelligence},
volume = {170},
pages = {114186},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.114186},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626004677},
author = {Alina Ciocarlan and Sylvie {Le Hégarat-Mascle} and Sidonie Lefebvre},
}
```
## Acknowledgments

This repo is built on yolov7 repo: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
