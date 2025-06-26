# 🛠️ `utils` — Utilities for Pretraining and Dataset Handling

This folder contains utility modules used throughout the project, including custom callbacks, configuration enums, and helper functions.

---

## 📦 Contents

### `__init__.py`

Convenience imports for:

- `get_class_list`
- `Signal` *(likely deprecated or unused)*

---

### `callbacks.py`

Custom model checkpoint logic for saving both full models and backbone-only weights:

- **`ModelAndBackboneCheckpoint`**
  - Inherits from `pytorch_lightning.callbacks.ModelCheckpoint`.
  - Saves the full model normally.
  - Additionally saves the backbone weights separately, useful for downstream transfer learning or modular pipelines.

---

### `enums.py`

Centralized configuration and model selection enums:

#### ✅ Backbone and Architecture

- `BackboneArchitecture`: e.g., `RESNET`, `VIT`, `XCIT`
- `BackboneSize`: e.g., `RESNET_18`, `VIT_BASE`, `XCIT_NANO_12`
- `BackboneProvider`: Implementation source, e.g., `TIMM`, `TORCHSIG`
- `BackboneSpec`: Combines architecture and size for precise specification.
- `BackboneType`: Common predefined combinations for easy selection.

#### ✅ Collation and Model Types

- `CollateType`: Defines single-view or multi-view collation logic.
- `SSLModelType`: Supported self-supervised learning models (BYOL, DINO, DenseCL, MoCoV3) with corresponding collation requirements.

#### ✅ Dataset and Transform Types

- `DatasetType`: Supported datasets including TorchSig and custom IQDM datasets.
- `TransformType`: Signal domain for transformations (`SPECTROGRAM` or `IQ`).

#### ✅ Metadata and Training Stages

- `MetadataModelType`: Currently supports `"mlp"` for the metadata tower.
- `TrainingStage`: Pipeline stages (`ssl`, `metadata`, `fusion`).

---

### `utils.py`

General-purpose helper functions:

- `get_class_list(config)`
  - Returns class or family label list depending on `config.family`.

**Legacy Note:** Contains a `Signal` class that may no longer be in use.

---

## ⚙️ Notes

- The enums in `enums.py` standardize experiment configuration and avoid hardcoding strings.
- `ModelAndBackboneCheckpoint` enables modular weight saving for models with separate backbones (e.g., two-tower architectures).
- This folder provides no direct executable scripts but underpins many higher-level modules.