# XCiT1D Backbone

This folder contains the customized 1D implementation of the XCiT (Cross-Covariance Image Transformer) architecture for processing IQ signal data.

## Features

- Adaptation of XCiT to 1D IQ signal inputs.
- Positional encoding applied along the time dimension.
- Two downsampling methods:
  - `ConvDownSampler`: Learnable convolutional downsampling.
  - `Chunker`: Simple average pooling-based downsampling.
- Returns both a classification token and a pooled token.
- Supports **masked pooling**, where only signal regions (based on masks) contribute to the pooled embedding.
- Fully compatible with contrastive learning pipelines like MoCo-v3.

## File Overview

- `xcit1d.py`: Full XCiT1D model definition with downsampling, positional encoding, and masked pooling support.

## Notes

- The XCiT model is loaded from `timm`, with patch embedding and head layers modified for 1D use.
- Masked pooling enables signal-focused learning when signal presence masks are available.