# Self-Supervised Learning (SSL) Models

This folder contains implementations of self-supervised learning (SSL) models for RF signal representation learning using PyTorch Lightning and Lightly.

## Available Models

| Model    | Maintained | Dataset Compatibility | Notes                                 |
|----------|------------|-----------------------|----------------------------------------|
| `BYOL`   | ❌ Legacy   | Old dataset only      | Not updated for masked pooling |
| `DINO`   | ❌ Legacy   | Old dataset only      | Not updated for masked pooling |
| `DenseCL`| ❌ Legacy   | Old dataset only      | Not updated for masked pooling |
| `MoCoV3` | ✅ Current  | Latest dataset (with time masks) | Fully compatible with masked pooling |

## Notes

- Only the `MoCoV3` model is currently compatible with the latest dataset format supporting time masks for masked pooling.
- Other models are provided for reference but may not function correctly with recent data processing changes.
- All models support optional online linear evaluation using Lightly's `OnlineLinearClassifier`.

## Dependencies

These models rely on:
- PyTorch Lightning
- Lightly
- TorchSig

## Usage

The models are integrated with the training pipelines defined under `selfrf/pretraining`.
