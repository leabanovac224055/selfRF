# selfRF

Self-Supervised Learning Framework for Radio Frequency Signals 

This repository contains a modular library for developing, training, and evaluating self-supervised learning (SSL) models for RF signal representation learning. The framework supports contrastive learning, masked pooling, and multimodal fusion of IQ data and metadata.

---

## Installation

**Install Dependencies**

```bash
# Install torchsig
pip install git+https://github.com/TorchDSP/torchsig.git

# Install detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install this package in development mode
pip install -e .
```

---

## Features

✅ Modular SSL pipeline for RF signals  
✅ Support for IQ-domain and spectrogram-based transforms  
✅ Predefined models: BYOL, DINO, DenseCL, MoCoV3  
✅ Late fusion of metadata via two-tower architecture  
✅ Masked pooling for variable-length signal handling  
✅ Data preprocessing scripts for SigMF datasets  
✅ Visualization utilities for IQ, spectrograms, and masks 

---

## Project Structure

```
selfrf/              # Main library (data loading, models, transforms, utils)
tools/               # Training & evaluation scripts
notebooks/           # Interactive exploration (focus: ssl_transforms.ipynb)
```

Every subfolder in `selfrf/` contains its own `README.md` with more details.

---

## Usage

### 1. SSL Pretraining (IQ Data Only)
Run pretraining on narrowband IQ data with contrastive learning:

```bash
python tools/iq_pretraining.py \
    --dataset IQDM_NARROWBAND \
    --num-epochs 10 \
    --batch-size 16 \
    --to-float-32 true \
    --ssl-model moco_v3 \
    --backbone xcit_tiny_12 \
    --backbone-provider timm \
    --online-linear-eval true
```

---

### 2. Metadata Tower Training
Train a model on precomputed metadata feature vectors:

```bash
python tools/meta_tower_training.py \
    --dataset TWO_TOWER_NARROWBAND \
    --batch-size 16 \
    --num-epochs 10
```

---

### 3. Fusion Model Training (IQ + Metadata)
Train a fusion model that combines IQ data and metadata:

```bash
python tools/fusion_train.py \
    --dataset TWO_TOWER_NARROWBAND \
    --batch-size 64 \
    --num-epochs 200 \
    --backbone resnet50 \
    --ssl-model moco_v3
```

---

## Evaluation

### 1. IQ Data Model Evaluation
Evaluate a pretrained IQ data model (replace `<path_to_checkpoint>` with your `.ckpt` path):

```bash
python tools/evaluate_pretraining.py \
    --model-path <path_to_checkpoint> \
    --backbone resnet50 \
    --dataset IQDM_NARROWBAND
```

---

### 2. Metadata Model Evaluation
Evaluate a trained metadata tower model:

```bash
python tools/evaluate_meta_tower.py \
    --model-path <path_to_checkpoint> \
    --dataset TWO_TOWER_NARROWBAND
```

---

### 3. Fusion Model Evaluation
Evaluate a trained fusion model:

```bash
python tools/evaluate_two_tower.py \
    --model-path <path_to_checkpoint> \
    --dataset TWO_TOWER_NARROWBAND \
    --backbone resnet50
```

---

## Notes

- The dataset argument must match the model type:
  - `IQDM_NARROWBAND` for IQ-only models
  - `TWO_TOWER_NARROWBAND` for Metadata or Fusion models
- For more details on specific components, check the `README.md` files in subfolders like `selfrf/models`, `selfrf/transforms`, and `selfrf/pretraining`.
- Only `moco_v3` is currently adapted for time-masked pooling. Other transforms or models may not support this.

---

## Todos for this project:

- [ ] Extend time-masked pooling to other SSL models if needed 
