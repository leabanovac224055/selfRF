"""TorchSig legacy Transforms from v0.6.0
"""

from typing import Callable, Optional
import numpy as np

from torchsig.signals.signal_types import Signal, SignalMetadata, DatasetSignal
from torchsig.transforms.dataset_transforms import DatasetTransform
from torchsig.transforms import functional as F
from torchsig.utils.dsp import torchsig_complex_data_type
from selfrf.transforms.extra.torchsig_legacy_utils import (
    get_distribution,
    NumericParameter,
    IntParameter,
    FloatParameter,
)

import selfrf.transforms.extra.torchsig_legacy_functional as F_LEGACY

__all__ = [
    "Identity",
    "RandomTimeShift",
    "AmplitudeReversal",
    "RandomFrequencyShift",
    "SpectrogramRandomResizeCrop",
    "SpectrogramPatchShuffle",
    "SpectrogramTranslation",
    "SpectrogramMosaicDownsample",
]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class Identity(DatasetTransform):
    """Just passes the data -- surprisingly useful in pipelines
    """

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        return signal


class RandomTimeShift(DatasetTransform):
    """Shifts signal in the time dimension by shift samples. 
    Zero-padding is applied to maintain input size.

    Args:
        shift: Amount to shift (samples):
            * If Callable, produces a sample by calling shift()
            * If int or float, shift is fixed at the value provided
            * If list, shift is any element in the list
            * If tuple, shift is in range of (tuple[0], tuple[1])
        min_time: Minimum time (fraction of signal) to maintain in view
    """

    def __init__(
        self,
        shift: NumericParameter = (-10, 10),
        min_time: float = .05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shift = get_distribution(shift, np.random.RandomState())
        self.min_time = min_time

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply time shift to signal."""
        # Get shift amount
        shift = self.shift()

        # Check bounds to ensure signals remain in view
        shift = self._check_time_bounds(signal, shift)

        signal.data = np.ascontiguousarray(
            F_LEGACY.time_shift(signal.data, int(shift)))

        self._transform_metadata(signal, shift)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal

    def _check_time_bounds(self, signal: DatasetSignal, shift: float) -> float:
        """
        Check new start and stop times to ensure signal is not cropped out of view
        """
        stop_shift_list = []
        start_shift_list = []
        total_samples = len(signal.data)

        for meta in signal.metadata:
            # Calculate normalized start/stop positions
            start_in_samples = meta.start_in_samples if hasattr(
                meta, 'start_in_samples') else int(meta.start * total_samples)
            duration_in_samples = meta.duration_in_samples if hasattr(
                meta, 'duration_in_samples') else int(meta.duration * total_samples)

            start_norm = start_in_samples / total_samples
            stop_norm = (start_in_samples +
                         duration_in_samples) / total_samples

            # Calculate potential new positions
            temp_start = start_norm + shift / total_samples
            temp_stop = stop_norm + shift / total_samples

            if temp_start > 1.0 or temp_stop < 0.0:
                if temp_start > 1.0:
                    new_shift = (1 - self.min_time -
                                 start_norm) * total_samples
                    start_shift_list.append(new_shift)
                else:
                    new_shift = (0 + self.min_time - stop_norm) * total_samples
                    stop_shift_list.append(new_shift)
            else:
                start_shift_list.append(shift)
                stop_shift_list.append(shift)

        if len(start_shift_list) == 0:
            start_shift_list = stop_shift_list

        if len(stop_shift_list) == 0:
            stop_shift_list = start_shift_list

        try:
            min_shift = np.max(
                (np.min(start_shift_list), np.min(stop_shift_list)))
            max_shift = np.min(
                (np.max(start_shift_list), np.max(stop_shift_list)))
            return np.random.uniform(min_shift, max_shift)
        except:
            return shift

    def _transform_metadata(self, signal: DatasetSignal, shift: float) -> None:
        """Update metadata after time shift."""
        total_samples = len(signal.data)

        valid_metadata = []
        for meta in signal.metadata:
            # Get start position in samples
            if hasattr(meta, 'start_in_samples'):
                meta.start_in_samples += int(shift)
                # Ensure values stay within bounds
                meta.start_in_samples = max(
                    0, min(total_samples - 1, meta.start_in_samples))

                # Update normalized start if it exists
                if hasattr(meta, 'start'):
                    meta.start = meta.start_in_samples / total_samples

                # Update duration if needed to keep within bounds
                if meta.start_in_samples + meta.duration_in_samples > total_samples:
                    meta.duration_in_samples = total_samples - meta.start_in_samples

                    # Update normalized duration if it exists
                    if hasattr(meta, 'duration'):
                        meta.duration = meta.duration_in_samples / total_samples
            else:
                # Working with normalized values
                meta.start += shift / total_samples
                meta.start = max(0.0, min(1.0, meta.start))

                # Update duration if needed
                if meta.start + meta.duration > 1.0:
                    meta.duration = 1.0 - meta.start

            # Only keep metadata if duration is positive
            if (hasattr(meta, 'duration_in_samples') and meta.duration_in_samples > 0) or \
               (hasattr(meta, 'duration') and meta.duration > 0):
                valid_metadata.append(meta)

        signal.metadata = valid_metadata


class AmplitudeReversal(DatasetTransform):
    """Applies an amplitude reversal to the input tensor by applying a value of
    -1 to each sample. Effectively the same as a static phase shift of pi
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply amplitude reversal to signal."""
        # Apply the transformation to the signal data
        signal.data = F_LEGACY.amplitude_reversal(signal.data)
        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class RandomFrequencyShift(DatasetTransform):
    """Shifts each signal in frequency by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling freq_shift()
            * If int or float, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])
    """

    def __init__(self, freq_shift: NumericParameter = (-0.5, 0.5), **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq_shift = get_distribution(freq_shift, np.random.RandomState())

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply frequency shift to signal."""
        # Get shift amount
        freq_shift = self.freq_shift()

        # Check bounds to ensure frequencies stay within valid range
        freq_shift = self._check_freq_bounds(signal, freq_shift)

        # Apply frequency shift
        signal.data = np.ascontiguousarray(
            F_LEGACY.freq_shift(signal.data, freq_shift))

        # Update metadata
        self._transform_metadata(signal, freq_shift)

        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal

    def _check_freq_bounds(self, signal: DatasetSignal, freq_shift: float) -> float:
        """
        Ensure frequency shift doesn't cause aliasing by keeping frequencies
        within the [-0.5, 0.5] range.
        """
        # First, apply a basic constraint to the frequency shift
        # This prevents extreme values immediately
        constrained_shift = max(-1.0, min(1.0, freq_shift))

        valid_shifts = []

        for meta in signal.metadata:
            try:
                # Check if proposed shift would move frequencies out of bounds
                test_lf = meta.lower_freq + constrained_shift
                test_hf = meta.upper_freq + constrained_shift

                if test_lf < -0.5:
                    # Calculate shift that would put lower_freq exactly at -0.5
                    safe_shift = -0.5 - meta.lower_freq
                    valid_shifts.append(safe_shift)
                elif test_hf > 0.5:
                    # Calculate shift that would put upper_freq exactly at 0.5
                    safe_shift = 0.5 - meta.upper_freq
                    valid_shifts.append(safe_shift)
                else:
                    # Current shift is fine
                    valid_shifts.append(constrained_shift)
            except Exception:
                # If any calculation errors occur, use the constrained original shift
                valid_shifts.append(constrained_shift)

        # If no valid shifts were calculated, return the constrained original
        if not valid_shifts:
            return constrained_shift

        # Find the shift closest to 0.0 (minimum change)
        valid_shifts = np.array([s for s in valid_shifts if -1.0 <= s <= 1.0])
        if len(valid_shifts) == 0:
            return constrained_shift

        return valid_shifts[np.abs(valid_shifts).argmin()]

    def _transform_metadata(self, signal: DatasetSignal, freq_shift: float) -> None:
        """Update metadata after frequency shift."""
        for meta in signal.metadata:
            # Only update metadata with frequency fields
            # Update the frequency information
            meta.lower_freq += freq_shift
            meta.upper_freq += freq_shift

            # Update bandwidth and center frequency if they exist
            meta.bandwidth = meta.upper_freq - meta.lower_freq

            meta.center_freq = meta.lower_freq + (meta.bandwidth / 2)


class SpectrogramRandomResizeCrop(DatasetTransform):
    """The SpectrogramRandomResizeCrop transforms the input IQ data into a
    spectrogram with a randomized FFT size and overlap. This randomization in
    the spectrogram computation results in spectrograms of various sizes. The
    width and height arguments specify the target output size of the transform.
    To get to the desired size, the randomly generated spectrogram may be
    randomly cropped or padded in either the time or frequency dimensions. This
    transform is meant to emulate the Random Resize Crop transform often used
    in computer vision tasks.

    Args:
        nfft (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The number of FFT bins for the random spectrogram.
            * If Callable, nfft is set by calling nfft()
            * If int, nfft is fixed by value provided
            * If list, nfft is any element in the list
            * If tuple, nfft is in range of (tuple[0], tuple[1])
            Defaults to (256, 1024).
        overlap_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The ratio of the (nfft-1) value to use as the overlap parameter for
            the spectrogram operation. Setting as ratio ensures the overlap is
            a lower value than the bin size.
            * If Callable, nfft is set by calling overlap_ratio()
            * If float, overlap_ratio is fixed by value provided
            * If list, overlap_ratio is any element in the list
            * If tuple, overlap_ratio is in range of (tuple[0], tuple[1])
            Defaults to (0.0, 0.2).
        detrend (Optional[str], optional): 
            _description_. Defaults to "constant".
        scaling (Optional[str], optional): 
            _description_. Defaults to "density".
        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.
        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'complex'.
        width (:obj:`int`):
            Target output width (time) of the spectrogram. Defaults to 512.
        height (:obj:`int`):
            Target output height (frequency) of the spectrogram. Defaults to 512.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly sample NFFT size in range [128,1024] and randomly crop/pad output spectrogram to (512,512)
        >>> transform = ST.SpectrogramRandomResizeCrop(nfft=(128,1024), overlap_ratio=(0.0,0.2), width=512, height=512)

    """

    def __init__(
        self,
        nfft: IntParameter = (256, 1024),
        overlap_ratio: FloatParameter = (0.0, 0.2),
        detrend: Optional[str] = "constant",
        scaling: Optional[str] = "density",
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = "complex",
        width: int = 512,
        height: int = 512,
    ) -> None:
        super(SpectrogramRandomResizeCrop, self).__init__()
        self.nfft = to_distribution(nfft, self.random_generator)
        self.overlap_ratio = to_distribution(
            overlap_ratio, self.random_generator)
        self.detrend: Optional[str] = None if detrend is None else detrend
        self.scaling: Optional[str] = None if scaling is None else scaling
        self.window_fcn = window_fcn
        self.mode = mode
        self.width = width
        self.height = height
        self.string = (
            self.__class__.__name__
            + "("
            + "nfft={}, ".format(nfft)
            + "overlap_ratio={}, ".format(overlap_ratio)
            + "detrend={}".format(self.detrend)
            + "scaling={}".format(self.scaling)
            + "window_fcn={}, ".format(window_fcn)
            + "mode={}, ".format(mode)
            + "width={}, ".format(width)
            + "height={}".format(height)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.nfft(), self.overlap_ratio())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        nfft, overlap_ratio = params
        nfft = int(nfft)
        nperseg = nfft
        noverlap = int(overlap_ratio * (nfft - 1))

        # First, perform the random spectrogram operation
        spec_data = F.spectrogram(
            signal["data"]["samples"],
            nperseg,
            noverlap,
            nfft,
            self.detrend,
            self.scaling,
            self.window_fcn,
            self.mode,
        )
        if self.mode == "complex":
            spec_data = self.spec_to_complex(spec_data)

        # Next, perform the random cropping/padding
        channels, curr_height, curr_width = spec_data.shape
        pad_height, crop_height = False, False
        pad_width, crop_width = False, False
        pad_height_samps, pad_width_samps = 0, 0
        if curr_height < self.height:
            pad_height = True
            pad_height_samps = self.height - curr_height
        elif curr_height > self.height:
            crop_height = True
        if curr_width < self.width:
            pad_width = True
            pad_width_samps = self.width - curr_width
        elif curr_width > self.width:
            crop_width = True

        if pad_height or pad_width:
            pad_height_start = np.random.randint(0, pad_height_samps // 2 + 1)
            pad_height_end = pad_height_samps - pad_height_start + 1
            pad_width_start = np.random.randint(0, pad_width_samps // 2 + 1)
            pad_width_end = pad_width_samps - pad_width_start + 1

            if self.mode == "complex":
                spec_data = self.pad_spec_complex(
                    spec_data,
                    self.pad_func,
                    pad_height_start,
                    pad_height_end,
                    pad_width_start,
                    pad_width_end,
                )
            else:
                spec_data = self.pad_spec(
                    self.pad_func,
                    pad_height_start,
                    pad_height_end,
                    pad_width_start,
                    pad_width_end,
                )

        crop_width_start = np.random.randint(
            0, max(1, curr_width - self.width))
        crop_height_start = np.random.randint(
            0, max(1, curr_height - self.height))
        spec_data = spec_data[
            :,
            crop_height_start: crop_height_start + self.height,
            crop_width_start: crop_width_start + self.width,
        ]
        signal["data"]["samples"] = spec_data

        # Update SignalMetadata
        new_meta = []
        for meta in signal["metadata"]:
            meta = meta_bound_frequency(meta)

            # Update labels based on padding/cropping
            if pad_height:
                meta = meta_pad_height(
                    meta, curr_height, self.height, pad_height_start)

            if crop_height:
                if (
                    meta["lower_freq"] + 0.5
                ) * curr_height >= crop_height_start + self.height or (
                    meta["upper_freq"] + 0.5
                ) * curr_height <= crop_height_start:
                    continue
                meta = self.meta_crop_height(
                    curr_height, crop_height_start, meta)

            if pad_width:
                meta = self.meta_pad_width(curr_width, pad_width_start, meta)

            if crop_width:
                if (
                    meta["start"] * curr_width >= crop_width_start + self.width
                    or meta["stop"] * curr_width <= crop_width_start
                ):
                    continue
                self.meta_crop_width(curr_width, crop_width_start, meta)

            # Append SignalMetadata to list
            new_meta.append(meta)

        signal["metadata"] = new_meta
        return signal

    def pad_func(self, vector, pad_width, iaxis, kwargs):
        vector[: pad_width[0]] = (
            np.random.rand(len(vector[: pad_width[0]])) * kwargs["pad_value"]
        )
        vector[-pad_width[1]:] = (
            np.random.rand(len(vector[-pad_width[1]:])) * kwargs["pad_value"]
        )

    def meta_crop_width(self, curr_width, crop_width_start, meta):
        if meta["start"] * curr_width <= crop_width_start:
            meta["start"] = 0.0
        else:
            meta["start"] = (meta["start"] * curr_width -
                             crop_width_start) / self.width

        if meta["stop"] * curr_width >= crop_width_start + self.width:
            meta["stop"] = 1.0
        else:
            meta["stop"] = (meta["stop"] * curr_width -
                            crop_width_start) / self.width
        meta["duration"] = meta["stop"] - meta["start"]

    def meta_crop_height(self, curr_height, crop_height_start, meta):
        if (meta["lower_freq"] + 0.5) * curr_height <= crop_height_start:
            meta["lower_freq"] = -0.5
        else:
            meta["lower_freq"] = (
                (meta["lower_freq"] + 0.5) * curr_height - crop_height_start
            ) / self.height - 0.5
        if (meta["upper_freq"] + 0.5) * curr_height >= crop_height_start + self.height:
            meta["upper_freq"] = crop_height_start + self.height
        else:
            meta["upper_freq"] = (
                (meta["upper_freq"] + 0.5) * curr_height - crop_height_start
            ) / self.height - 0.5
        meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
        meta["center_freq"] = meta["lower_freq"] + meta["bandwidth"] / 2
        return meta

    def meta_pad_width(
        self, curr_width, pad_width_start, meta: SignalMetadata
    ) -> SignalMetadata:
        meta["start"] = (meta["start"] * curr_width +
                         pad_width_start) / self.width
        meta["stop"] = (meta["stop"] * curr_width +
                        pad_width_start) / self.width
        meta["duration"] = meta["stop"] - meta["start"]
        return meta

    def pad_spec(
        self, pad_func, pad_height_start, pad_height_end, pad_width_start, pad_width_end
    ):
        spec_data = np.pad(
            spec_data,
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            min_value=np.percentile(np.abs(spec_data[0]), 50),
        )

        return spec_data

    def pad_spec_complex(
        self,
        spec_data,
        pad_func,
        pad_height_start,
        pad_height_end,
        pad_width_start,
        pad_width_end,
    ):
        new_data_real = np.pad(
            spec_data[0],
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            pad_value=np.percentile(np.abs(spec_data[0]), 50),
        )
        new_data_imag = np.pad(
            spec_data[1],
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            pad_value=np.percentile(np.abs(spec_data[1]), 50),
        )
        spec_data = np.concatenate(
            [
                np.expand_dims(new_data_real, axis=0),
                np.expand_dims(new_data_imag, axis=0),
            ],
            axis=0,
        )
        return spec_data

    def spec_to_complex(self, spec_data):
        new_tensor = np.zeros(
            (2, spec_data.shape[0], spec_data.shape[1]), dtype=np.float32
        )
        new_tensor[0, :, :] = np.real(spec_data).astype(np.float32)
        new_tensor[1, :, :] = np.imag(spec_data).astype(np.float32)
        spec_data = new_tensor
        return spec_data


class SpectrogramPatchShuffle(DatasetTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `PatchShuffle Regularization <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            patch_size sets the size of each patch to shuffle
            * If Callable, produces a sample by calling patch_size()
            * If int or float, patch_size is fixed at the value provided
            * If list, patch_size is any element in the list
            * If tuple, patch_size is in range of (tuple[0], tuple[1])

        shuffle_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If Callable, produces a sample by calling shuffle_ratio()
            * If int or float, shuffle_ratio is fixed at the value provided
            * If list, shuffle_ratio is any element in the list
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        patch_size: NumericParameter = (2, 16),
        shuffle_ratio: FloatParameter = (0.01, 0.10),
    ) -> None:
        super(SpectrogramPatchShuffle, self).__init__()
        self.patch_size = to_distribution(patch_size, self.random_generator)
        self.shuffle_ratio = to_distribution(
            shuffle_ratio, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "patch_size={}, ".format(patch_size)
            + "shuffle_ratio={}".format(shuffle_ratio)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.patch_size(), self.shuffle_ratio())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        patch_size, shuffle_ratio = params
        signal["data"]["samples"] = F_LEGACY.spec_patch_shuffle(
            signal["data"]["samples"], patch_size, shuffle_ratio
        )
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        return signal


class SpectrogramTranslation(DatasetTransform):
    """Transform that inputs a spectrogram and applies a random time/freq
    translation

    Args:
         time_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            time_shift sets the translation along the time-axis
            * If Callable, produces a sample by calling time_shift()
            * If int, time_shift is fixed at the value provided
            * If list, time_shift is any element in the list
            * If tuple, time_shift is in range of (tuple[0], tuple[1])

        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            freq_shift sets the translation along the freq-axis
            * If Callable, produces a sample by calling freq_shift()
            * If int, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        time_shift: IntParameter = (-128, 128),
        freq_shift: IntParameter = (-128, 128),
    ) -> None:
        super().__init__()
        self.time_shift = to_distribution(time_shift, self.random_generator)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "time_shift={}, ".format(time_shift)
            + "freq_shift={}".format(freq_shift)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.time_shift(), self.freq_shift())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        time_shift, freq_shift = params
        signal["data"]["samples"] = F_LEGACY.spec_translate(
            signal["data"]["samples"], time_shift, freq_shift
        )
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:

        time_shift, freq_shift = params
        new_meta = []
        for meta in signal["metadata"]:
            # Update time fields
            meta["start"] = (
                meta["start"] + time_shift / signal["data"]["samples"].shape[1]
            )
            meta["stop"] = (
                meta["stop"] + time_shift / signal["data"]["samples"].shape[1]
            )
            if meta["start"] >= 1.0 or meta["stop"] <= 0.0:
                continue
            meta["start"] = 0.0 if meta["start"] < 0.0 else meta["start"]
            meta["stop"] = 1.0 if meta["stop"] > 1.0 else meta["stop"]
            meta["duration"] = meta["stop"] - meta["start"]

            # Trim any out-of-capture freq values
            meta = meta_bound_frequency(meta)

            # Update freq fields
            meta["lower_freq"] = (
                meta["lower_freq"] + freq_shift /
                signal["data"]["samples"].shape[2]
            )
            meta["upper_freq"] = (
                meta["upper_freq"] + freq_shift /
                signal["data"]["samples"].shape[2]
            )
            if meta["lower_freq"] >= 0.5 or meta["upper_freq"] <= -0.5:
                continue

            meta = meta_bound_frequency(meta)

            # Append SignalMetadata to list
            new_meta.append(meta)

        # Set output data's SignalMetadata to above list
        signal["metadata"] = new_meta
        return signal
