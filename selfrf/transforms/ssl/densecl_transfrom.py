from typing import Tuple

from torch import Tensor

import torchvision.transforms.v2 as T

from torchsig.signals import DatasetSignal
from torchsig.transforms.base_transforms import RandomApply, Compose
from torchsig.transforms.dataset_transforms import (
    TimeReversal,
    SpectralInversionDatasetTransform,
    DatasetTransform,
)

from selfrf.transforms.extra.transforms import MultiViewTransform, RandomAWGN, SpectrogramImageHighQuality


class PyTorchImageTransformWrapper(DatasetTransform):
    """Wrapper to apply PyTorch image transforms to TorchSig spectrogram outputs.

    This adapter sits between a TorchSig spectrogram transform and PyTorch image transforms,
    ensuring proper data format conversion.
    """

    def __init__(self, torch_transform, **kwargs):
        super().__init__(**kwargs)
        self.torch_transform = torch_transform

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        data = signal.data

       # add channel dimension if not present
        if data.ndim == 2:
            data = data.unsqueeze(0)

        # throw error if data is not a 3D tensor
        if data.ndim != 3:
            raise ValueError("Data must be a  [C, H, W] tensor.")

        data = self.torch_transform(data)
        # Convert back to DatasetSignal
        signal.data = data
        # Ensure the signal is still a DatasetSignal
        self.update(signal)
        return signal


class DenseCLTransform(MultiViewTransform):

    def __init__(
        self,
        nfft: int = 512,
        tensor_transform: DatasetTransform = SpectrogramImageHighQuality,
        min_scale: float = 0.2,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        noise_power_db: Tuple = (-30, 20),
        **kwargs,
    ):
        view_transform = DenseCLViewTransform(
            nfft=nfft,
            spectrogram_transform=tensor_transform,
            min_scale=min_scale,
            tr_prob=hf_prob,
            si_prob=vf_prob,
            noise_power_db=noise_power_db,
        )
        super().__init__(transforms=[view_transform, view_transform])


class DenseCLViewTransform:
    def __init__(
        self,
        nfft: int = 512,
        spectrogram_transform: DatasetTransform = SpectrogramImageHighQuality(
            nfft=512),
        min_scale: float = 0.2,
        tr_prob: float = 0.5,
        si_prob: float = 0.5,
        noise_power_db: Tuple = (-30, 20),
    ):

        transform = [
            RandomAWGN(noise_power_db),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            spectrogram_transform,
            PyTorchImageTransformWrapper(
                T.RandomResizedCrop(size=nfft, scale=(min_scale, 1.0))
            ),
        ]
        self.transform = Compose(transform)

    def __call__(self, signal: DatasetSignal) -> Tensor:

        return self.transform(signal)
