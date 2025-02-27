
from typing import Any, Literal, Sequence
from copy import deepcopy
import numpy as np
import torch

from torchsig.signals.signal_types import Signal, DatasetSignal
from torchsig.transforms.base_transforms import Transform, Compose
from torchsig.transforms.dataset_transforms import DatasetTransform
from torchsig.transforms.transform_utils import (
    get_distribution,
    NumericParameter
)
from copy import deepcopy
import numpy as np

from . import functional as F

__all__ = [
    "MultiViewTransform",
    "SpectrogramImage",
    "AmplitudeScale",
    "ToDtype",
    "ToTensor",
    "ToSpectrogramTensor",
]


class MultiViewTransform(Transform):
    """Transforms a signal into multiple views."""

    def __init__(self, transforms: Sequence[Compose]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        """Creates independent views with separate data copies and returns all views"""
        views = []
        for transform in self.transforms:
            # Create fresh copy for each transform pipeline
            data_copy = deepcopy(signal)
            transformed_view = transform(data_copy)
            views.append(transformed_view)

            # For object-based signals, create a new Signal-like object
        result = deepcopy(signal)
        result.data = [view.data for view in views]
        return result


class AmplitudeScale(DatasetTransform):
    """Scales the amplitude of the input tensor

    Args:
        scale (:py:class:`~Callable`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The scaling factor to apply.
            * If Callable, produces a sample by calling scale()
            * If float, scale is fixed at the value provided
            * If list, scale is any element in the list
            * If tuple, scale is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Fixed scale of 2.0
        >>> transform = ST.AmplitudeScale(2.0)
        >>> # Random scale between 0.5 and 2.0
        >>> transform = ST.AmplitudeScale((0.5, 2.0))
    """

    def __init__(
        self,
        scale: NumericParameter = (0.5, 2.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.scale = get_distribution(scale, self.random_generator)
        self.string = f"{self.__class__.__name__}(scale={scale})"

    def parameters(self) -> tuple:
        return (float(self.scale()),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        scale_value = params[0]
        signal["data"]["samples"] = F.amplitude_scale(
            signal["data"]["samples"], scale_value)
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
        self.string = f"{self.__class__.__name__}(dtype={dtype})"

    def parameters(self) -> tuple:
        """
        Returns:
            A tuple containing the chosen dtype.
        """
        return (self.dtype(),)

    def transform_data(self, signal: dict, params: tuple) -> dict:
        """
        Converts the 'samples' in signal["data"] to the desired dtype.

        Returns:
            The modified signal dictionary.
        """
        chosen_dtype = params[0]
        # Use astype to convert the NumPy array to the new dtype.
        signal["data"]["samples"] = signal["data"]["samples"].astype(
            chosen_dtype)
        return signal


class ToTensor(DatasetTransform):
    """Converts a numpy array to a PyTorch tensor.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ToTensor()

    """

    def __init__(
        self,
        to_float_32: bool = False
    ) -> None:
        super().__init__()
        self.to_float_32 = to_float_32

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:

        # convert to torch tensor
        tensor = torch.from_numpy(signal.data)

        # convert to float32 if requested
        if self.to_float_32:
            tensor = tensor.float()

        # add channel dimension
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
