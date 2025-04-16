from typing import Optional, Tuple
import numpy as np
from torch import Tensor

from torchsig.signals import DatasetSignal
from torchsig.transforms.base_transforms import Transform, RandomApply, Compose
from torchsig.transforms.dataset_transforms import (
    TimeReversal,
    SpectralInversionDatasetTransform,
    ComplexTo2D,
    DatasetTransform,
    CutOut,
    RandomMagRescale,
)


from ..extra import torchsig_legacy_transforms as T_LEGACY
from ..extra import MultiViewTransform, RandomAWGN


class BYOLView1Transform(Transform):
    def __init__(
        self,
        max_time_shift: int = 500,
        max_freq_shift: float = 0.15,
        tr_prob: float = 0.5,
        si_prob: float = 0.5,
        noise_power_db: tuple = (-100, -100),
        cutout_duration: tuple = (0.05, 0.1),
        min_amplitude_scale: float = 0.5,
        max_amplitude_scale: float = 2,
        max_phase_shift_rad: float = np.pi/4,
        tensor_transform: DatasetTransform = ComplexTo2D(),
    ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(allow_spectral_inversion=False), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(
                duration=cutout_duration,
                cut_type=["zeros"],
            ),
            RandomMagRescale(
                scale=(min_amplitude_scale, max_amplitude_scale),
            ),
            RandomAWGN(noise_power_db=noise_power_db),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tensor:

        return self.transform(signal)


class BYOLView2Transform(Transform):
    def __init__(
        self,
        max_time_shift: int = 500,
        max_freq_shift: float = 0.15,
        tr_prob: float = 0.5,
        si_prob: float = 0.5,
        noise_power_db: tuple = (-100, -80),
        cutout_duration: tuple = (0.05, 0.1),
        min_amplitude_scale: float = 0.5,
        max_amplitude_scale: float = 2,
        max_phase_shift_rad: float = np.pi/4,
        tensor_transform: DatasetTransform = ComplexTo2D(),
    ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversal(allow_spectral_inversion=False), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(
                duration=cutout_duration,
                cut_type=["zeros"],
            ),
            RandomMagRescale(
                scale=(min_amplitude_scale, max_amplitude_scale),
            ),
            RandomAWGN(noise_power_db=noise_power_db),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
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
        **kwargs,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform(
            tensor_transform=tensor_transform,
        )

        view_2_transform = view_2_transform or BYOLView2Transform(
            tensor_transform=tensor_transform,
        )

        super().__init__(transforms=[view_1_transform, view_2_transform])
