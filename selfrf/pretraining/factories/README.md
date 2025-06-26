
# 🏭 Factories — Model, Dataset, Collate, and Transform Builders

This folder contains modular factory classes and utility functions responsible for dynamically constructing core components of the RF learning pipeline based on the provided configuration. The factory pattern allows flexible experimentation and clean separation of concerns across model building, dataset creation, and data transformations.

---

## 📦 Folder Structure

```
factories/
├── __init__.py               # Import shortcuts for key factory functions
├── collate_fn_factory.py      # Builds collate functions for different training stages
├── dataset_factory.py         # Dynamically builds dataloaders for TorchSig or IQDM datasets
├── model_factory.py           # Builds backbones, SSL models, metadata towers, and fusion heads
└── transform_factory.py       # Constructs data and target transforms based on config
```

---

## ⚒️ Main Factory Components

### ✅ `model_factory.py`
- Builds backbone models for IQ or spectrogram inputs (e.g., ResNet, XCiT, ViT).
- Constructs self-supervised learning models like BYOL, DINO, MoCo-v3.
- Provides metadata tower construction (e.g., MLP) for two-tower models.
- Builds fusion heads for late fusion of IQ and metadata embeddings.

### ✅ `dataset_factory.py`
- Dynamically creates appropriate dataloaders for:
  - **TorchSig datasets** (narrowband, wideband).
  - **Custom IQDM datasets**, including TwoTower datasets for multimodal learning.
- Automatically injects transforms, target transforms, and collate functions.

### ✅ `collate_fn_factory.py`
- Provides correct collate logic based on:
  - SSL model type (single/multi-view).
  - Two-tower training stage.
  - Metadata-only training.
  - Evaluation setups.
- Ensures proper IQ, mask, metadata, and label handling for each stage.

### ✅ `transform_factory.py`
- Builds data transforms for IQ or spectrogram pipelines.
- Wraps augmentations for self-supervised learning methods (BYOL, DINO, etc.).
- Creates target transforms (e.g., class index extraction) for supervised tasks.

---

## 🛠️ Example Usage

These factories are not run directly. They are imported and used throughout the pipeline, for example:

```python
from selfrf.pretraining.factories import (
    build_backbone, build_ssl_model, build_dataloader, build_transform
)

# Build backbone
backbone = build_backbone(config)

# Build dataloader with correct transforms
datamodule = build_dataloader(config)

# Build SSL model
model = build_ssl_model(config)
```

---

## 📌 Notes

- The factories are **config-driven** — behavior depends on arguments from `TrainingConfig` or `EvaluationConfig`.
- Extendable architecture — new models, datasets, or transforms can be registered easily by updating the corresponding registry.