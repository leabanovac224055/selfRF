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

- fix bbox labels and spectrogram
- update number of classes in vitdet
- vitdet evaluation
