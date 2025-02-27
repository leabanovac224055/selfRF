"""TorchSig legacy Transforms from v0.6.0
"""

from typing import Callable, Optional
from scipy import signal as sp
import numpy as np

from torchsig.signals.signal_types import Signal, SignalMetadata, DatasetSignal
from torchsig.transforms.dataset_transforms import DatasetTransform
from torchsig.transforms import functional as F

import selfrf.transforms.extra.torchsig_legacy_functional as F_LEGACY
from .torchsig_legacy_functional import (
    FloatParameter,
    IntParameter,
    NumericParameter,
    to_distribution,
)

__all__ = [
    "Identity",
    "RandomPhaseShift",
    "RandomTimeShift",
    "TimeCrop",
    "AmplitudeReversal",
    "AmplitudeScale",
    "RandomFrequencyShift",
    "SpectrogramRandomResizeCrop",
    "SpectrogramPatchShuffle",
    "SpectrogramTranslation",
    "SpectrogramMosaicDownsample",
    "SpectrogramImage",
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
    """Shifts tensor in the time dimension by shift samples. Zero-padding is applied to maintain input size.

    Args:
        shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling shift()
            * If int or float, shift is fixed at the value provided
            * If list, shift is any element in the list
            * If tuple, shift is in range of (tuple[0], tuple[1])

        interp_rate (:obj:`int`):
            Interpolation rate used by internal interpolation filter

        taps_per_arm (:obj:`int`):
            Number of taps per arm used in filter. More is slower, but more accurate.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Shift inputs by range of (-10, 20) samples with uniform distribution
    """

    def __init__(
        self,
        shift: NumericParameter = (-10, 10),
        interp_rate: int = 100,
        taps_per_arm: int = 24,
        min_time: float = .05,
        **kwargs,
    ) -> None:
        super(RandomTimeShift, self).__init__(**kwargs)
        self.shift = to_distribution(shift, self.random_generator)
        self.interp_rate = interp_rate
        self.min_time = min_time
        num_taps = int(taps_per_arm * interp_rate)

        self.taps = (
            sp.firwin(num_taps, 1.0 / interp_rate, width=1.0 /
                      (interp_rate / 4.0), scale=True)
            * interp_rate
        )
        self.string = (
            self.__class__.__name__
            + "("
            + "shift={}, ".format(shift)
            + "interp_rate={}, ".format(interp_rate)
            + "taps_per_arm={}".format(taps_per_arm)
            + ")"
        )
        self.applied_transform_name = "RandomTimeShift"

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply time shift to signal."""
        signal = self.transform_data(signal)
        signal = self.transform_meta(signal)

        # Record the applied transform
        for meta in signal.metadata:
            if not hasattr(meta, "applied_transforms"):
                meta.applied_transforms = []
            meta.applied_transforms.append(self.applied_transform_name)

        return signal

    def check_time_bounds(self, signal: DatasetSignal) -> float:
        """
        Method checks new start and stop times to ensure signal is not cropped out
        of view
        """
        stop_shift_list = []
        start_shift_list = []
        shift = self.shift
        total_samples = len(signal.data)

        for meta in signal.metadata:
            # Calculate normalized start/stop positions
            start_norm = meta.start_in_samples / total_samples
            stop_norm = (meta.start_in_samples +
                         meta.duration_in_samples) / total_samples

            # Calculate potential new positions
            temp_start = start_norm + shift / total_samples
            temp_stop = stop_norm + shift / total_samples

            if temp_start > 1.0 or temp_stop < 0.0:
                if temp_start > 1.0:
                    new_shift = (1 - self.min_time -
                                 start_norm) * total_samples
                    start_shift_list.append(new_shift)
                else:
                    new_shift = (1 - self.min_time - stop_norm) * total_samples
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
        except:
            return (shift,)

        return (np.random.uniform(min_shift, max_shift, 1)[0],)

    def transform_data(self, signal: DatasetSignal, params: tuple) -> DatasetSignal:
        params_new = self.check_time_bounds(signal, params)
        integer_part, _ = divmod(params_new[0], 1)
        integer_time_shift: int = int(integer_part) if integer_part else 0

        signal.data = F_LEGACY.time_shift(signal.data, integer_time_shift)
        return signal

    def transform_meta(self, signal: DatasetSignal, params: tuple) -> DatasetSignal:

        params_new = self.check_time_bounds(signal, params)
        shift = params_new[0]
        total_samples = len(signal.data)

        valid_metadata = []
        for meta in signal.metadata:
            # Update start position
            meta.start_in_samples += int(shift)

            # Ensure values stay within bounds
            meta.start_in_samples = max(
                0, min(total_samples - 1, meta.start_in_samples))

            # Update duration if needed to keep within bounds
            if meta.start_in_samples + meta.duration_in_samples > total_samples:
                meta.duration_in_samples = total_samples - meta.start_in_samples

            # Only keep metadata if duration is positive
            if meta.duration_in_samples > 0:
                valid_metadata.append(meta)

        signal.metadata = valid_metadata
        if len(signal.metadata) == 0:
            print("Warning: empty metadata after RandomTimeShift!")

        return signal


class TimeCrop(DatasetTransform):
    """Crops a tensor in the time dimension to the specified length. Optional
    crop techniques include: start, center, end, & random

    Args:
        crop_type (:obj:`str`):
            Type of cropping to perform. Options are: `start`, `center`, `end`,
            and `random`. `start` crops the input tensor such that the first
            `length` samples are returned. `center` crops the input tensor such
            that the center `length` samples are returned. `end` crops the
            input tensor such that the last `length` samples are returned.
            `random` crops randomly in the range `[0,length-1]`.

        length (:obj:`int`):
            Number of samples to include.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Crop inputs to first 256 samples
        >>> transform = ST.TimeCrop(crop_type='start', length=256)
        >>> # Crop inputs to center 512 samples
        >>> transform = ST.TimeCrop(crop_type='center', length=512)
        >>> # Crop inputs to last 1024 samples
        >>> transform = ST.TimeCrop(crop_type='end', length=1024)
        >>> # Randomly crop any 2048 samples from input
        >>> transform = ST.TimeCrop(crop_type='random', length=2048)

    """

    def __init__(
        self,
        crop_type: str = "random",
        crop_length: int = 256,
        signal_length: int = 1024,
    ) -> None:
        super(TimeCrop, self).__init__()
        self.crop_type = crop_type
        self.crop_length = crop_length
        self.signal_length = signal_length
        if self.crop_type not in ("start", "center", "end", "random"):
            raise ValueError(
                "Crop type must be: `start`, `center`, `end`, or `random`")

        self.string = (
            self.__class__.__name__
            + "("
            + "crop_type={}, ".format(crop_type)
            + "length={}".format(crop_length)
            + ")"
        )

    def parameters(self) -> tuple:
        if self.crop_type == "start":
            start = 0
        elif self.crop_type == "end":
            start = self.signal_length - self.crop_length
        elif self.crop_type == "center":
            start = (self.signal_length - self.crop_length) // 2
        elif self.crop_type == "random":
            start = np.random.randint(0, self.signal_length - self.crop_length)

        return start, self.crop_length

    def check_time_bounds(self, signal: Signal, params: tuple) -> float:
        """
            Method checks new start and stop times to ensure signal is not cropped out
            of view
        """
        start_list = []
        start, crop_length = params

        for meta in signal["metadata"]:
            original_start_sample = meta["start"] * \
                data_shape(signal["data"])[0]
            original_stop_sample = meta["stop"] * data_shape(signal["data"])[0]
            # new_start_sample = original_start_sample - start
            new_start_sample = start
            new_stop_sample = original_stop_sample - start
            start_clip = np.clip(
                float(new_start_sample / crop_length), a_min=0.0, a_max=1.0)
            stop_clip = np.clip(
                float(new_stop_sample / crop_length), a_min=0.0, a_max=1.0)
            duration = stop_clip - start_clip
            if duration < .001:
                start_list.append(
                    int(meta["start"] * data_shape(signal["data"])[0]))
            else:
                start_list.append(int(new_start_sample))

        return np.min(start_list), crop_length

    def transform_data(self, signal: Signal, params: tuple) -> Signal:

        if len(signal["metadata"]) == 0:
            return signal

        if len(signal["data"]["samples"]) == self.crop_length:
            return signal

        params = self.check_time_bounds(signal, params)
        # if signal["metadata"][0]["num_samples"] < self.crop_length:
        if data_shape(signal["data"])[0] < self.crop_length:
            raise ValueError(
                "Input data length {} is less than requested length {}".format(
                    data_shape(signal["data"])[0], self.crop_length
                )
            )

        signal["data"]["samples"] = F_LEGACY.time_crop(
            signal["data"]["samples"], params[0], self.crop_length)
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:

        params = self.check_time_bounds(signal, params)
        start, crop_length = params
        for meta in signal["metadata"]:
            original_start_sample = meta["start"] * \
                data_shape(signal["data"])[0]
            original_stop_sample = meta["stop"] * data_shape(signal["data"])[0]
            new_start_sample = original_start_sample - start
            new_stop_sample = original_stop_sample - start
            meta["start"] = np.clip(
                float(new_start_sample / crop_length), a_min=0.0, a_max=1.0)
            meta["stop"] = np.clip(
                float(new_stop_sample / crop_length), a_min=0.0, a_max=1.0)
            meta["duration"] = meta["stop"] - meta["start"]
            meta["num_samples"] = crop_length

        return signal


class AmplitudeReversal(DatasetTransform):
    """Applies an amplitude reversal to the input tensor by applying a value of
    -1 to each sample. Effectively the same as a static phase shift of pi

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.amplitude_reversal(
            signal["data"]["samples"])
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
        super(AmplitudeScale, self).__init__(**kwargs)
        self.scale = to_distribution(scale, self.random_generator)
        self.string = f"{self.__class__.__name__}(scale={scale})"

    def parameters(self) -> tuple:
        return (float(self.scale()),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        scale_value = params[0]
        signal["data"]["samples"] = F_LEGACY.amplitude_scale(
            signal["data"]["samples"], scale_value)
        return signal


class RandomFrequencyShift(DatasetTransform):
    """Shifts each tensor in freq by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling freq_shift()
            * If int or float, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = ST.RandomFrequencyShift(freq_shift=(-0.25, 0.25))
        >>> # Frequency shift inputs always fs/10
        >>> transform = ST.RandomFrequencyShift(freq_shift=0.1)
        >>> # Frequency shift inputs with normal distribution with stdev .1
        >>> transform = ST.RandomFrequencyShift(freq_shift=lambda size: np.random.normal(0, .1, size))
        >>> # Frequency shift inputs with either -fs/4 or fs/4 (discrete)
        >>> transform = ST.RandomFrequencyShift(freq_shift=[-.25, .25])

    """

    def __init__(self, freq_shift: NumericParameter = (-0.5, 0.5), **kwargs) -> None:
        super(RandomFrequencyShift, self).__init__(**kwargs)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.string = (
            self.__class__.__name__ +
            "(" + "freq_shift={}".format(freq_shift) + ")"
        )

    def parameters(self) -> tuple:
        return (self.freq_shift(),)

    def check_freq_bounds(self, signal: Signal, freq_shift: float) -> float:
        """
            Method checks frequency mins and maxes and adjust the new_rate to ensure
            frequency bounds stay within the +-.5 boundary.
        """
        ret_list = []
        for meta in signal["metadata"]:
            test_lf = meta["lower_freq"] + freq_shift
            test_hf = meta["upper_freq"] + freq_shift
            if test_lf < -.5 or test_hf > .5:
                if test_lf < -.5:
                    new_shift = -.5 - meta['lower_freq']
                else:
                    new_shift = .5 - meta['upper_freq']
                ret_list.append(new_shift)
            else:
                ret_list.append(freq_shift)
        return find_nearest(ret_list, 0.)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        freq_shift = self.check_freq_bounds(signal, params[0])
        signal["data"]["samples"] = F_LEGACY.freq_shift(
            signal["data"]["samples"], freq_shift)

        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        freq_shift = self.check_freq_bounds(signal, params[0])
        for meta in signal["metadata"]:
            # Check bounds for partial signals
            meta["lower_freq"] += freq_shift
            meta["upper_freq"] += freq_shift
            meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
            meta["center_freq"] = meta["lower_freq"] + meta["bandwidth"] * 0.5

        return signal


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
        super(SpectrogramTranslation, self).__init__()
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


class SpectrogramImage(DatasetTransform):
    """Transforms SignalData to spectrogram image

    Args:
        None


    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.SpectrogramImage() 

    """

    def __init__(
        self,
    ) -> None:
        super(SpectrogramImage, self).__init__()
        self.string: str = (
            self.__class__.__name__
        )

    def transform_data(self, signal: Signal) -> Signal:
        signal["data"]["samples"] = F_LEGACY.spectrogram_image(
            signal["data"]["samples"],
        )
        return signal
