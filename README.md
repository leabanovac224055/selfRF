# selfRF

Library for Self-Supervised Learning for Radio Frequency Signals

## Installation

### Install Dependencies

```bash
# Install torchsig
pip install git+https://github.com/TorchDSP/torchsig.git

# Install detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install this package in development mode
pip install -e .
```

## Training

### SSL Pretraining

```bash
python tools/pretraining.py --num-epochs 10 --to-float-32 true --backbone resnet50 --backbone-provider timm  --online-linear-eval true
```

## Todos for this project:

- update number of classes in vitdet

## Results

### SSL Pretraining

| Model    | Method | Backbone | Dataset    | Epochs | Batch Size | KNN Eval |
| -------- | ------ | -------- | ---------- | ------ | ---------- | -------- |
| ResNet50 | BYOL   | ResNet50 | Narrowband | 200    | 128        | 86.5     |

### Object Detection Finetuning

| Model        | Backbone | Pretraining | AP    | AP50  | AP75  | APs   | APm   | APl   |
| ------------ | -------- | ----------- | ----- | ----- | ----- | ----- | ----- | ----- |
| Faster R-CNN | ResNet50 | BYOL        | 37.88 | 68.60 | 44.42 | 21.88 | 42.20 | 35.19 |
