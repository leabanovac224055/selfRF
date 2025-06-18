from typing import Optional, Dict, Tuple
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
from torchsig.utils.random import Seedable
from ..extra import torchsig_legacy_transforms as T_LEGACY
from ..extra import MultiViewTransform, RandomAWGN


class RandomTimeShiftWithMask(Transform, Seedable):
    def __init__(self, shift_range):
        super().__init__()
        self.shift_range = shift_range
        self.children = []  # required for Seedable mechanics

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        shift_amount = np.random.randint(self.shift_range[0], self.shift_range[1] + 1)

        signal.data = np.roll(signal.data, shift_amount, axis=-1)

        if hasattr(signal, "time_mask"):
            signal.time_mask = np.roll(signal.time_mask, shift_amount)

        return signal
    

class TimeReversalWithMask(Transform, Seedable):
    def __init__(self, allow_spectral_inversion=False):
        super().__init__()
        self.allow_spectral_inversion = allow_spectral_inversion
        self.children = []  # needed for Seedable mechanics

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        # Reverse time axis for IQ data
        signal.data = signal.data[..., ::-1]

        # Reverse mask if present
        if hasattr(signal, "time_mask"):
            signal.time_mask = signal.time_mask[::-1]

        return signal
    
    
class MoCoView1Transform(Transform):
    """
    Strong Augmentation - More aggressive transformations for contrastive learning.
    """
    def __init__(
        self,
        transform_params: Dict[str, float],
        tensor_transform: DatasetTransform = ComplexTo2D(),
    ) -> None:
        super().__init__()

        # Unpack the dynamic parameters for the strong augmentation
        max_time_shift = transform_params.get("max_time_shift", 805)
        max_freq_shift = transform_params.get("max_freq_shift", 0.3985574086980508)
        tr_prob = transform_params.get("tr_prob", 0.608450454556431)
        si_prob = transform_params.get("si_prob", 0.7762550830719575)
        noise_power_db = transform_params.get("noise_power_db", (-75.22313643070225, -62.118486958653406))
        cutout_duration = transform_params.get("cutout_duration", (0.04555615917235689, 0.17060771871824668))
        min_amplitude_scale = transform_params.get("min_amplitude_scale", 0.7624829965779709)
        max_amplitude_scale = transform_params.get("max_amplitude_scale", 2.0610869794590556)
        max_phase_shift_rad = transform_params.get("max_phase_shift_rad", 1.1226980361020837)

        # Compose the list of strong transformations
        transforms = [
            RandomTimeShiftWithMask((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversalWithMask(allow_spectral_inversion=False), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(duration=cutout_duration, cut_type=["zeros"]),
            RandomMagRescale(scale=(min_amplitude_scale, max_amplitude_scale)),
            RandomAWGN(noise_power_db=noise_power_db),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tuple[Tensor, Tensor]:
        signal = self.transform(signal)
        return signal.data, signal.time_mask


class MoCoView2Transform(Transform):
    """
    Weak Augmentation - Milder transformations for contrastive learning.
    """
    def __init__(
        self,
        transform_params: Dict[str, float],
        tensor_transform: DatasetTransform = ComplexTo2D(),
    ) -> None:
        super().__init__()

        # Unpack the dynamic parameters for the weak augmentation
        max_time_shift = transform_params.get("max_time_shift", 387)
        max_freq_shift = transform_params.get("max_freq_shift", 0.18816041163381997)
        tr_prob = transform_params.get("tr_prob", 0.4608719907109846)
        si_prob = transform_params.get("si_prob", 0.43061635232409606)
        noise_power_db = transform_params.get("noise_power_db", (-101.52797781738239, -73.12024758794661))
        cutout_duration = transform_params.get("cutout_duration", (0.028072889881628003, 0.06292169619335006))
        min_amplitude_scale = transform_params.get("min_amplitude_scale", 0.7678840019601538)
        max_amplitude_scale = transform_params.get("max_amplitude_scale", 1.742623699317833)
        max_phase_shift_rad = transform_params.get("max_phase_shift_rad", 0.9994746248630416)

        # Compose the list of weak transformations
        transforms = [
            RandomTimeShiftWithMask((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            RandomApply(TimeReversalWithMask(allow_spectral_inversion=False), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(duration=cutout_duration, cut_type=["zeros"]),
            RandomMagRescale(scale=(min_amplitude_scale, max_amplitude_scale)),
            RandomAWGN(noise_power_db=noise_power_db),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = Compose(transforms=transforms)

    def __call__(self, signal: DatasetSignal) -> Tuple[Tensor, Tensor]:
        signal = self.transform(signal)
        return signal.data, signal.time_mask


class MoCoTransform(MultiViewTransform):
    def __init__(
        self,
        view_1_params: Optional[Dict[str, float]] = None,
        view_2_params: Optional[Dict[str, float]] = None,
        tensor_transform: DatasetTransform = ComplexTo2D(),
        **kwargs,
    ):
        # Initialize with dynamic parameters for each view
        view_1_params = view_1_params or {}
        view_2_params = view_2_params or {}

        # Strong and weak views
        view_1_transform = MoCoView1Transform(view_1_params, tensor_transform)
        view_2_transform = MoCoView2Transform(view_2_params, tensor_transform)

        super().__init__(transforms=[view_1_transform, view_2_transform])