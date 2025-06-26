# Model Architectures

This folder contains all model components used within the project, organized by modality and purpose.

## Folder Structure

- **`iq_models/`**  
  Backbones for processing raw IQ signal data. Includes ResNet1D and XCiT1D architectures.  
  _[See: `iq_models/README.md`]_

- **`meta_models/`**  
  Lightweight models for processing metadata feature vectors (e.g., center frequency, bandwidth, duration) and fusion heads for combining IQ and metadata embeddings.  
  _[See: `meta_models/README.md`]_

- **`spectrogram_models/`**  
  Models designed to process spectrogram representations of signals. Currently used with DenseCL pipelines.

- **`ssl_models/`**  
  Self-supervised learning models built on top of the backbones. Includes BYOL, DINO, DenseCL, and MoCoV3.  
  _[See: `ssl_models/README.md`]_

## Notes

- Backbones follow a unified interface, producing consistent embeddings for downstream tasks.
- Some backbones (e.g., XCiT) support masked pooling for applications like masked contrastive learning.
- Fusion heads and metadata towers enable multimodal learning by combining IQ data with structured metadata.

