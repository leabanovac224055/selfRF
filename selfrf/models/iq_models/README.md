# IQ Models

This module provides backbone architectures designed to process raw IQ (In-phase and Quadrature) signal data for radio frequency (RF) machine learning tasks. These backbones serve as feature extractors for self-supervised learning (SSL) pipelines and other downstream tasks.

## Available Architectures

- **ResNet1D**  
  Adaptation of standard 2D ResNet models for 1D IQ signal inputs. See [`resnet/`](./resnet) for details.

- **XCiT1D**  
  Custom implementation of the XCiT transformer architecture adapted for 1D IQ signals, with support for masked pooling. See [`xcit/`](./xcit) for details.

## Usage

These models are instantiated through factory functions within the codebase and are compatible with both contrastive SSL pipelines and downstream evaluation tasks.