# ResNet1D Backbone

This folder contains the ResNet1D implementation used for IQ signal processing in the project. It leverages `timm`'s 2D ResNet models, converted for 1D input via `convert_2d_model_to_1d`.

## Features

- Supports multiple ResNet variants (e.g., ResNet18, ResNet34, ResNet50).
- Processes IQ signal data with two input channels (real and imaginary).
- Optionally outputs raw features for downstream use (feature extractor mode).
- Simple wrapper (`ResNetWrapper`) ensures compatibility with masked pooling pipelines.

## File Overview

- `resnet1d.py`: Core function to construct 1D ResNet models.
- `resnet_wrapper.py`: Wraps ResNet to provide a standardized `(cls_token, pooled_token)` output interface.

## Example

See the `build_resnet1d` function for model instantiation details.