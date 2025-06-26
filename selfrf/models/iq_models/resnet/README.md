# ResNet1D Backbone

This folder contains the ResNet1D implementation for IQ signal processing.

## Overview

ResNet1D models are adapted from standard 2D ResNet architectures using `timm` and converted for 1D IQ signal inputs. A dedicated wrapper ensures these models **do not** apply masked pooling, keeping their behavior consistent with traditional CNN feature extraction.

## Features

- Supports ResNet variants like ResNet18, ResNet34, ResNet50.
- Processes IQ signals with two input channels (real and imaginary).
- The `ResNetWrapper` explicitly avoids masked pooling to maintain CNN-like behavior.
- Provides `(cls_token, pooled_token)` output format for compatibility with contrastive learning pipelines.

## File Overview

- `resnet1d.py`  
  Contains the logic to construct 1D ResNet models from `timm`.

- `resnet_wrapper.py`  
  Wraps ResNet models to produce a consistent output format `(features, None)`  
  Masked pooling is deliberately excluded for ResNet backbones.

## Why No Masked Pooling?

ResNet is a convolutional architecture where masked pooling is not necessary for proper feature extraction. Masked pooling is reserved for transformer-based backbones (like XCiT) where sequence-wise aggregation requires selective pooling.

## Example

See the `build_resnet1d` function for how to instantiate 1D ResNet models for this project.