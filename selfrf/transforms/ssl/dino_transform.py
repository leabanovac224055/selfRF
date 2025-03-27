from typing import Optional
import numpy as np
from torch import Tensor

from torchsig.signals import DatasetSignal
from torchsig.transforms.base_transforms import Transform, RandomApply, Compose
from torchsig.transforms.dataset_transforms import (
    TimeReversal, SpectralInversionDatasetTransform,
    ComplexTo2D, DatasetTransform)

from ..extra import torchsig_legacy_transforms as T_LEGACY
from ..extra import MultiViewTransform, RandomAWGN


class DINOView1Transform(Transform):
    """Global View: Easier for Teacher"""

    def __init__(self,
                 max_time_shift: int = 400,
                 max_freq_shift: float = 0.2,
                 tr_prob: float = 0.5,
                 si_prob: float = 0.5,
                 noise_power_db: tuple = (-20, 10),
                 min_amplitude_scale: float = -4,
                 max_amplitude_scale: float = 4,
                 max_phase_shift_rad: float = np.pi/8,
                 tensor_transform: DatasetTransform = ComplexTo2D()):
        super().__init__()
        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            RandomAWGN(noise_power_db),
            tensor_transform,
        ]
        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:
        return self.transform(signal)


class DINOView2Transform(Transform):
    """Local View: Harder for Student"""

    def __init__(self,
                 max_time_shift: int = 1500,
                 max_freq_shift: float = 0.5,
                 tr_prob: float = 0.2,
                 si_prob: float = 0.2,
                 noise_power_db: tuple = (-40, 5),
                 min_amplitude_scale: float = -12,
                 max_amplitude_scale: float = 12,
                 max_phase_shift_rad: float = np.pi/4,
                 tensor_transform: DatasetTransform = ComplexTo2D()):
        super().__init__()
        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            RandomAWGN(noise_power_db),
            tensor_transform,
        ]
        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:
        return self.transform(signal)


class AdaptiveDINOViewTransform(Transform):
    """Adaptive Local View (Smooth Scaling Based on Similarity)."""

    def __init__(self, similarity: float, tensor_transform: DatasetTransform = ComplexTo2D()):
        """Smoothly scales the augmentation difficulty based on similarity."""

        # Define min & max ranges for each augmentation
        time_shift_range = (500, 2000)  # Min: 500, Max: 2000
        freq_shift_range = (0.2, 0.6)   # Min: 0.2, Max: 0.6
        noise_power_range = (-20, -50)  # Min: -20dB, Max: -50dB

        # Smoothly scale each parameter
        max_time_shift = int(
            time_shift_range[0] + (time_shift_range[1] - time_shift_range[0]) * (1 - similarity))
        max_freq_shift = freq_shift_range[0] + (
            freq_shift_range[1] - freq_shift_range[0]) * (1 - similarity)
        noise_power_db = (
            noise_power_range[0] + (noise_power_range[1] - noise_power_range[0]) * (1 - similarity), 5)

        print(
            f"[Adaptive Augmentations] Similarity: {similarity:.3f} → TimeShift: {max_time_shift}, FreqShift: {max_freq_shift:.2f}, Noise: {noise_power_db}")

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomAWGN(noise_power_db),
            tensor_transform,
        ]
        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:
        return self.transform(signal)


class DINOTransform(MultiViewTransform):
    """DINO Multi-View Transform (Supports Adaptive Augmentation)."""

    def __init__(
            self,
            similarity: Optional[float] = None,
            view_1_transform: Optional[DINOView1Transform] = None,
            view_2_transform: Optional[DINOView2Transform] = None,
            tensor_transform: DatasetTransform = ComplexTo2D(),
            **kwargs,
    ):

        # Standard global view (Easy for teacher)
        view_1_transform = view_1_transform or DINOView1Transform(
            tensor_transform=tensor_transform)

        # **Fix:** Use similarity to decide which view to apply
        if similarity is not None:
            view_2_transform = AdaptiveDINOViewTransform(
                similarity=similarity, tensor_transform=tensor_transform)
        else:
            view_2_transform = view_2_transform or DINOView2Transform(
                tensor_transform=tensor_transform)

        super().__init__(transforms=[view_1_transform, view_2_transform])
