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
                 max_time_shift: int = 300,                  # Slightly easier than BYOL
                 max_freq_shift: float = 0.08,               # Slightly easier
                 tr_prob: float = 0.3,                      # Lower prob for easier view
                 si_prob: float = 0.3,
                 noise_power_db: tuple = (-100, -100),       # Lighter noise
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
                 max_time_shift: int = 800,                  # Harder than global
                 max_freq_shift: float = 0.20,               # Harder than global
                 tr_prob: float = 0.6,                      # More aggressive
                 si_prob: float = 0.6,
                 noise_power_db: tuple = (-100, -80),       # Stronger noise 
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
    

class DINOTransform(MultiViewTransform):
    """DINO Multi-View Transform (Two standard views: global for teacher, local for student)."""

    def __init__(
            self,
            view_1_transform: Optional[DINOView1Transform] = None,
            view_2_transform: Optional[DINOView2Transform] = None,
            tensor_transform: DatasetTransform = ComplexTo2D(),
            **kwargs,
    ):

        # Standard global view (Easy for teacher)
        view_1_transform = view_1_transform or DINOView1Transform(
            tensor_transform=tensor_transform)

        # Standard local view (Hard for student)    
        view_2_transform = view_2_transform or DINOView2Transform(
            tensor_transform=tensor_transform)

        super().__init__(transforms=[view_1_transform, view_2_transform])
