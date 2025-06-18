
import random
from typing import List, Sequence, Tuple, Union
from copy import deepcopy
import numpy as np
import torch

import torchaudio
from torchsig.signals.signal_types import DatasetSignal
from torchsig.transforms.base_transforms import Transform, Compose
from torchsig.transforms.dataset_transforms import DatasetTransform
import torchsig.transforms.functional as torchsig_F
from copy import deepcopy
import numpy as np

from . import functional as F

__all__ = [
    "RandomPrinter",
    "MultiViewTransform",
    "RandomAWGN",
    "AmplitudeScale",
    "ToDtype",
    "SpectrogramImageHighQuality",
    "ToTensor",
    "ToSpectrogramTensor",
]


class RandomPrinter(DatasetTransform):
    """Debug transform that prints random values from a distribution.

    Useful for testing if DataLoader workers are using different random seeds.

    Args:
        value_range: Range for random values (min, max)
        name: Name to identify this instance in debug outputs
    """

    def __init__(
        self,
        value_range: Union[List, Tuple] = (0, 1000),
        name: str = "debug",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.value_distribution = self.get_distribution(value_range)
        self.name = name

        # Get worker info for debugging
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "main"

        # Print initialization message
        print(f"RandomPrinter '{self.name}' initialized in worker {worker_id}")

        # Set instance counter to track call sequence
        self.counter = 0

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Sample from distribution and print value without modifying signal."""
        # Get random value from distribution
        value = self.value_distribution()

        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "main"

        # Increment counter
        self.counter += 1

        # Print debug info
        print(f"[Worker {worker_id}] {self.name} #{self.counter}: {value}")

        # Return unmodified signal
        return signal


class MultiViewTransform(Transform):
    """Transforms a signal into multiple views."""

    def __init__(self, transforms: Sequence[Compose]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Creates independent views with separate data copies and returns all views"""
        views = []
        for transform in self.transforms:
            # Create fresh copy for each transform pipeline
            data_copy = deepcopy(signal)
            transformed_view = transform(data_copy)
            views.append(transformed_view)

        return tuple(views)


class RandomAWGN(DatasetTransform):
    """
    Adds white Gaussian noise to a signal with a random noise power in the given range.

    Args:
        noise_power_db (:py:class:`~Callable`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The noise power in dB to apply.
            * If
                * Callable, produces a sample by calling noise_power_db()
                * float, noise_power_db is fixed at the value provided
                * list, noise_power_db is any element in the list
                * tuple, noise_power_db is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        noise_power_db: Union[List, Tuple] = (0, 20.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.noise_power_db_distribution = self.get_distribution(
            noise_power_db)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply random AWGN to the signal."""
        noise_power_db = self.noise_power_db_distribution()
        signal.data = torchsig_F.awgn(
            signal.data,
            noise_power_db=noise_power_db,
            rng=self.random_generator
        )
        # signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class AmplitudeScale(DatasetTransform):
    """Scales the amplitude of the input tensor

    Args:
        scale (:py:class:`~Callable`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The scaling factor to apply.
            * If Callable, produces a sample by calling scale()
            * If float, scale is fixed at the value provided
            * If list, scale is any element in the list
            * If tuple, scale is in range of (tuple[0], tuple[1])
    """

    def __init__(
        self,
        scale: Union[List, Tuple] = (0.5, 2.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.scale_distribution = self.get_distribution(scale)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        scale_value = self.scale_distribution()
        signal.data = F.amplitude_scale(signal.data, scale_value)
        self.update(signal)
        return signal


class ToDtype(DatasetTransform):
    """
    Transform that converts the 'samples' of a signal (a NumPy ndarray) to a specific dtype.
    """

    def __init__(
            self,
            dtype: np.dtype,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Converts the 'samples' in signal["data"] to the desired dtype.

        Returns:
            The modified signal dictionary.
        """
        # Use astype to convert the NumPy array to the new dtype.
        signal.data = signal.data.astype(self.dtype)
        self.update(signal)
        return signal


class ToTensor(DatasetTransform):
    """Converts a numpy array to a PyTorch tensor.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ToTensor()

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:

        # convert to torch tensor
        tensor = torch.from_numpy(signal.data)

        signal.data = tensor
        self.update(signal)
        return signal


class ToSpectrogramTensor(DatasetTransform):
    """Converts a numpy array to a PyTorch tensor to shape (C, X, Y),
    where C is the number of channels (1), X is the number of time steps and y is the number of frequency bins.
    """

    def __init__(
        self,
        to_float_32: bool = False
    ) -> None:
        super().__init__()
        self.to_float_32 = to_float_32

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        # check if data is in spectrogram format
        if len(signal.data.shape) != 2:
            raise ValueError("Data must be in spectrogram format (2D)")

        # Make sure the array is contiguous before converting to torch tensor
        # This fixes the negative stride issue
        if not signal.data.flags.c_contiguous:
            signal.data = np.ascontiguousarray(signal.data)

        # convert to torch tensor
        tensor = torch.from_numpy(signal.data)

        # convert to float32 if requested
        if self.to_float_32:
            tensor = tensor.float()

        # add channel dimension
        tensor = tensor.unsqueeze(0)

        signal.data = tensor
        self.update(signal)
        return signal


class SpectrogramImageHighQuality(DatasetTransform):
    """High-quality RF spectrogram transformation following research paper implementation.

    This transforms complex IQ data into a normalized spectrogram image optimized for
    self-supervised learning, using the same approach as the paper but for grayscale output.
    """

    def __init__(
        self,
        nfft: int = 512,
        db_scale: bool = True,
        normalize: bool = True,
        invert: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.nfft = nfft
        self.db_scale = db_scale
        self.normalize = normalize
        self.invert = invert

        # Create spectrogram transform once at init time
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft,
            win_length=self.nfft,
            hop_length=self.nfft,
            window_fn=torch.blackman_window,
            normalized=False,
            center=False,
            onesided=False,
            power=2,
        )

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Convert complex IQ data to grayscale spectrogram image."""
        # Convert to torch tensor
        data = torch.from_numpy(signal.data)

        # Apply spectrogram transform
        x = self.spectrogram(data)

        # Normalize by infinity norm as in paper
        if self.normalize:
            norm_val = torch.linalg.norm(x.flatten(), ord=float("inf"))
            x = x / (norm_val + 1e-12)

        # Apply FFT shift and flip for proper orientation
        x = torch.fft.fftshift(x, dim=0)
        x = torch.flip(x, dims=[0])  # same as flipud in the paper

        # Convert to dB scale
        if self.db_scale:
            x = 10 * torch.log10(x + 1e-12)

        # Linear scaling to [0,1] range using min-max values
        x_min = torch.min(x)
        x_max = torch.max(x)

        # Linear transform to map to [0,1] using the same method as the paper
        slope = 1.0 / (x_max - x_min + 1e-12)
        intercept = -x_min * slope
        x = x * slope + intercept

        # Invert colors if requested (common in RF visualization)
        if self.invert:
            x = 1.0 - x

        # add channel dimension
        x = x.unsqueeze(0)

        # Convert back to numpy array for compatibility with other transforms
        signal.data = x

        self.update(signal)
        return signal
