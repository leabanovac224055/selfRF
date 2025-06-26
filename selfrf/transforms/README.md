# Transforms

This folder contains custom data transforms for RF signal processing using TorchSig.

## Structure

- `extra/` — Additional transforms and TorchSig-compatible utilities  
- `ssl/` — Self-supervised learning (SSL) transforms for contrastive learning setups  

## Notes

- Some transforms in `ssl` (e.g., DINO, BYOL) may not be fully up to date with the latest dataset structures, especially regarding time masks used for masked pooling.  
- The `moco_transform` is currently the only SSL transform fully aligned with the latest dataset format supporting masked pooling.

## Usage

These transforms are designed to integrate with TorchSig's `DatasetSignal` objects and the wider self-supervised learning pipeline defined in this project.
