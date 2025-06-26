
# SSL Transforms for Multi-View Data Augmentation

This folder contains **multi-view data augmentation scripts** used for contrastive self-supervised learning (SSL) applied to IQ signals or spectrograms.

Each transform produces two differently augmented versions of the same signal, following standard contrastive learning principles.

---

## Available Transforms

| Transform         | Description                                                | Masked Pooling Support | Status |
|-------------------|------------------------------------------------------------|-------------------------|--------|
| `BYOLTransform`   | Two-view transform for BYOL-style contrastive learning.    | **Not supported**       | Outdated |
| `DINOTransform`   | Teacher-student global/local views for DINO training.      | **Not supported**       | Outdated |
| `DenseCLTransform`| Spectrogram-based transform for DenseCL training.          | **Not supported**       | Never actively used |
| `MoCoTransform`   | Two-view transform for MoCo v3-style contrastive learning. | ✅ Fully supported       | **Up-to-date** |

---

## ⚠️ Known Limitations

- Only **`MoCoTransform`** is updated for datasets containing `time_mask` fields used in masked pooling.
- The other transforms (`BYOLTransform`, `DINOTransform`, `DenseCLTransform`) are **not compatible** with new datasets containing masks.

---

## Structure

Each transform consists of:

✅ Two individual view transforms with independent augmentation parameters  
✅ A wrapper `MultiViewTransform` that applies both views and returns them as a tuple  
✅ Augmentations include time shifts, frequency shifts, inversion, CutOut, amplitude scaling, AWGN noise, and phase shifts  
✅ Spectrogram transforms apply additional image augmentations  

---

## Example Usage

```python
transform = MoCoTransform()
(view1_data, view1_mask), (view2_data, view2_mask) = transform(signal)
```

---

## Notes

- This folder depends on the helper utilities defined in `selfrf.transforms.extra`.
- Augmentations for spectrogram-based methods like DenseCL wrap standard TorchVision transforms.

---

## Maintenance Status

✅ `MoCoTransform` actively maintained  
⚠️ Other transforms remain unchanged and may require updates for new datasets  
