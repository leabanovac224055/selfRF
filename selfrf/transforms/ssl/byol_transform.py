from typing import Optional, Tuple
import numpy as np
from torch import Tensor

from torchsig.signals import DatasetSignal
from torchsig.transforms.base_transforms import Transform, RandomApply, Compose
from torchsig.transforms.dataset_transforms import (
    TimeReversal, SpectralInversionDatasetTransform,
    ComplexTo2D, DatasetTransform)

from ..extra import torchsig_legacy_transforms as T_LEGACY
from ..extra import MultiViewTransform, RandomAWGN


class BYOLView1Transform(Transform):
    def __init__(self,
                 max_time_shift: int = 500,
                 max_freq_shift: float = 0.35,
                 tr_prob: float = 0.5,
                 si_prob: float = 0.5,
                 noise_power_db: Tuple = (-30, 10),
                 min_amplitude_scale: float = -6,
                 max_amplitude_scale: float = 6,
                 max_phase_shift_rad: float = np.pi/4,
                 tensor_transform: DatasetTransform = ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            RandomAWGN(noise_power_db),
            # AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            # T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:

        return self.transform(signal)


class BYOLView2Transform(Transform):
    def __init__(self,
                 max_time_shift: int = 1000,
                 max_freq_shift: float = 0.35,
                 tr_prob: float = 0.3,
                 si_prob: float = 0.3,
                 noise_power_db: Tuple = (-30, 10),

                 min_amplitude_scale: float = -10,
                 max_amplitude_scale: float = 10,
                 max_phase_shift_rad: float = np.pi/8,
                 tensor_transform: DatasetTransform = ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            RandomAWGN(noise_power_db),
            # AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            # T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:

        return self.transform(signal)


class BYOLTransform(MultiViewTransform):
    def __init__(
        self,
        view_1_transform: Optional[BYOLView1Transform] = None,
        view_2_transform: Optional[BYOLView2Transform] = None,
        tensor_transform: DatasetTransform = ComplexTo2D(),
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform(
            tensor_transform=tensor_transform,
        )

        view_2_transform = view_2_transform or BYOLView2Transform(
            tensor_transform=tensor_transform,
        )

        super().__init__(transforms=[view_1_transform, view_2_transform])
