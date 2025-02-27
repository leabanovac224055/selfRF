"""Functional transforms from legacy torchsig repo v0.6.0
"""
from typing import Callable, List, Optional, Tuple, Union
from torchsig.utils.dsp import low_pass
from numba import njit
from scipy import signal as sp
from functools import partial
import numpy as np
import pywt
import os
import cv2

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


__all__ = [
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "uniform_discrete_distribution",
    "uniform_continuous_distribution",
    "to_distribution",
    "resample",
    "make_sinc_filter",
    "awgn",
    "impulsive_interference",
    "interleave_complex",
    "real",
    "imag",
    "complex_magnitude",
    "wrapped_phase",
    "discrete_fourier_transform",
    "continuous_wavelet_transform",
    "time_shift",
    "time_crop",
    "freq_shift",
    "freq_shift_avoid_aliasing",
    "_fractional_shift_helper",
    "fractional_shift",
    "amplitude_reversal",
    "amplitude_scale",
    "roll_off",
    "clip",
    "random_convolve",
    "drop_spec_samples",
    "spec_patch_shuffle",
    "spec_translate",
    "spectrogram_image",
]

FloatParameter = Union[Callable[[int], float],
                       float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def uniform_discrete_distribution(
    choices: List, random_generator: Optional[np.random.Generator] = None
):
    """Unifrom Discrete Distribution

    Args:
        choices (List): List of discrete variables to sample from.
        random_generator (Optional[np.random.Generator], optional): Random Generator to use. Defaults to None (new generator created internally)

    Returns:
        _type_: _description_
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
    lower: Union[int, float],
    upper: Union[int, float],
    random_generator: Optional[np.random.Generator] = None,
):
    """Uniform Continuous Distribution

    Args:
        lower (Union[int, float]): Lowest number possible in distribution.
        upper (Union[int, float]): Highest number possible in distribution.
        random_generator (Optional[np.random.Generator], optional): Random Generator to use. Defaults to None (new generator created internally)

    Returns:
        _type_: _description_
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    return partial(random_generator.uniform, lower, upper)


def to_distribution(
    param: Union[
        int,
        float,
        str,
        Callable,
        List[int],
        List[float],
        List[str],
        Tuple[int, int],
        Tuple[float, float],
    ],
    random_generator: Optional[np.random.Generator] = None,
):
    """Create Numpy Random Generator(s) over a distribution.

    Args:
        param (Union[ int, float, str, Callable, List[int], List[float], List[str], Tuple[int, int], Tuple[float, float], ]): Range, type, or variables specifying random distribution.
        random_generator (Optional[np.random.Generator], optional): Random generator to use. Defaults to None (new generator created internally).

    Returns:
        _type_: _description_
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    if isinstance(param, Callable):  # type: ignore
        return param

    if isinstance(param, list):
        #######################################################################
        # [BUG ALERT]: Nested tuples within lists does not function as desired.
        # Below will instantiate a random distribution from the list; however,
        # each call will only come from the previously randomized selection,
        # but the desired behavior would be for this to randomly select each
        # region at call time. Commenting out for now, but should revisit in
        # the future to add back the functionality.
        #######################################################################
        # if isinstance(param[0], tuple):
        #     tuple_from_list = param[random_generator.randint(len(param))]
        #     return uniform_continuous_distribution(
        #         tuple_from_list[0],
        #         tuple_from_list[1],
        #         random_generator,
        #     )
        return uniform_discrete_distribution(param, random_generator)

    if isinstance(param, tuple):
        return uniform_continuous_distribution(
            param[0],
            param[1],
            random_generator,
        )

    if isinstance(param, int) or isinstance(param, float):
        return uniform_discrete_distribution([param], random_generator)

    return param


@njit(cache=False)
def make_sinc_filter(beta, tap_cnt, sps, offset=0):
    """
    return the taps of a sinc filter
    """
    ntap_cnt = tap_cnt + ((tap_cnt + 1) % 2)
    t_index = np.arange(-(ntap_cnt - 1) // 2,
                        (ntap_cnt - 1) // 2 + 1) / np.double(sps)

    taps = np.sinc(beta * t_index + offset)
    taps /= np.sum(taps)

    return taps[:tap_cnt]


def awgn(tensor: np.ndarray, noise_power_db: float) -> np.ndarray:
    """Adds zero-mean complex additive white Gaussian noise with power of
    noise_power_db.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        noise_power_db (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_noise = np.random.randn(*tensor.shape)
    imag_noise = np.random.randn(*tensor.shape)
    return tensor + (10.0 ** (noise_power_db / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)


@njit(cache=False)
def impulsive_interference(
    tensor: np.ndarray,
    amp: float,
    per_offset: float,
) -> np.ndarray:
    """Applies an impulsive interferer to tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        amp (:obj:`float`):
            Maximum vector magnitude of complex interferer signal

        per_offset (:obj:`float`)
            Interferer offset into the tensor as expressed in a fraction of the tensor length.

    """
    beta = 0.3
    num_samps = len(tensor)
    sinc_pulse = make_sinc_filter(beta, num_samps, 0.1, 0)
    imp = amp * np.roll(sinc_pulse / np.max(sinc_pulse),
                        int(per_offset * num_samps))
    rand_phase = np.random.uniform(0, 2 * np.pi)
    imp = np.exp(1j * rand_phase) * imp
    output: np.ndarray = tensor + imp
    return output


def interleave_complex(tensor: np.ndarray) -> np.ndarray:
    """Converts complex vectors to real interleaved IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Interleaved vectors.
    """
    new_tensor = np.empty((tensor.shape[0] * 2,))
    new_tensor[::2] = np.real(tensor)
    new_tensor[1::2] = np.imag(tensor)
    return new_tensor


def real(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a real-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            real(tensor)
    """
    return np.real(tensor)


def imag(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a imaginary-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            imag(tensor)
    """
    return np.imag(tensor)


def complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a complex magnitude vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            abs(tensor)
    """
    return np.abs(tensor)


def wrapped_phase(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a wrapped-phase vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            angle(tensor)
    """
    return np.angle(tensor)


def discrete_fourier_transform(tensor: np.ndarray) -> np.ndarray:
    """Computes DFT of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            fft(tensor). normalization is 1/sqrt(n)
    """
    return np.fft.fft(tensor, norm="ortho")


def continuous_wavelet_transform(
    tensor: np.ndarray, wavelet: str, nscales: int, sample_rate: float
) -> np.ndarray:
    """Computes the continuous wavelet transform resulting in a Scalogram of the complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        wavelet (:obj:`str`):
            Name of the mother wavelet.
            If None, wavename = 'mexh'.

        nscales (:obj:`int`):
            Number of scales to use in the Scalogram.
            If None, nscales = 33.

        sample_rate (:obj:`float`):
            Sample rate of the signal.
            If None, fs = 1.0.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Scalogram of tensor along time dimension
    """
    scales = np.arange(1, nscales)
    cwtmatr, _ = pywt.cwt(
        tensor, scales=scales, wavelet=wavelet, sampling_period=1.0 / sample_rate
    )

    # if the dtype is complex then return the magnitude
    if np.iscomplexobj(cwtmatr):
        cwtmatr = abs(cwtmatr)

    return cwtmatr


def time_shift(tensor: np.ndarray, t_shift: int) -> np.ndarray:
    """Shifts tensor in the time dimension by tshift samples. Zero-padding is
    applied to maintain input size.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted.

        t_shift (:obj:`int` or :class:`numpy.ndarray`):
            Number of samples to shift right or left (if negative)

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor shifted in time of size tensor.shape
    """
    # Valid Range Error Checking
    if np.max(np.abs(t_shift)) >= tensor.shape[0]:
        return np.zeros_like(tensor, dtype=np.complex64)

    # This overwrites tensor as side effect, modifies inplace
    if t_shift > 0:
        tmp = tensor[:-t_shift]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (t_shift, 0), "constant", constant_values=0 + 0j)
    elif t_shift < 0:
        tmp = tensor[-t_shift:]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (0, -t_shift), "constant", constant_values=0 + 0j)
    return tensor


def time_crop(tensor: np.ndarray, start: int, length: int) -> np.ndarray:
    """Crops a tensor in the time dimension from index start(inclusive) for length samples.

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be cropped.

        start (:obj:`int` or :class:`numpy.ndarray`):
            index to begin cropping

        length (:obj:`int`):
            number of samples to include

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor cropped in time of size (tensor.shape[0], length)
    """
    # Type and Size checking
    if length < 0:
        raise ValueError("Length must be greater than 0")

    if np.any(start < 0):
        raise ValueError("Start must be greater than 0")

    if np.max(start) >= tensor.shape[0] or length == 0:
        return np.empty(shape=(1, 1))

    return tensor[start: start + length]


def freq_shift(tensor: np.ndarray, f_shift: float) -> np.ndarray:
    """Shifts each tensor in freq by freq_shift along the time dimension

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    sinusoid = np.exp(
        2j * np.pi * f_shift * np.arange(tensor.shape[0], dtype=np.float64)
    )
    mult = np.multiply(tensor, np.asarray(sinusoid))
    return mult


def freq_shift_avoid_aliasing(
    tensor: np.ndarray,
    f_shift: float,
) -> np.ndarray:
    """Similar to `freq_shift` function but performs the frequency shifting at
    a higher sample rate with filtering to avoid aliasing

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    # Match output size to input
    num_iq_samples = tensor.shape[0]

    # Interpolate up to avoid frequency wrap around during shift
    up = 2
    down = 1
    tensor = sp.resample_poly(tensor, up, down)

    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    tensor = convolve_out[lidx:ridx]

    # Freq shift to desired center freq
    time_vector = np.arange(tensor.shape[0], dtype=np.float64)
    tensor = tensor * np.exp(2j * np.pi * f_shift / up * time_vector)

    # Filter to remove out-of-band regions
    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - int(num_iq_samples * up)) // 2
    ridx = lidx + len(tensor)
    # prune to be correct size out of filter
    tensor = convolve_out[: int(num_iq_samples * up)]

    # Decimate back down to correct sample rate
    tensor = sp.resample_poly(tensor, down, up)

    return tensor[:num_iq_samples]


@njit(cache=False)
def _fractional_shift_helper(taps: np.ndarray, raw_iq: np.ndarray, stride: int, offset: int) -> np.ndarray:
    """Fractional shift. First, we up-sample by a large, fixed amount. Filter with 1/upsample_rate/2.0,
    Next we down-sample by the same, large fixed amount with a chosen offset. Doing this efficiently means not actually zero-padding.

    The efficient way to do this is to decimate the taps and filter the signal with some offset in the taps.
    """
    # We purposely do not calculate values within the group delay.
    group_delay = ((taps.shape[0] - 1) // 2 - (stride - 1)) // stride + 1
    if offset < 0:
        offset += stride
        group_delay -= 1

    # Decimate the taps.
    taps = taps[offset::stride]

    # Determine output size
    num_taps = taps.shape[0]
    num_raw_iq = raw_iq.shape[0]
    output = np.zeros(
        ((num_taps + num_raw_iq - 1 - group_delay),), dtype=np.complex128)

    # This is a just convolution of taps and raw_iq
    for o_idx in range(output.shape[0]):
        idx_mn = o_idx - (num_raw_iq - 1) if o_idx >= num_raw_iq - 1 else 0
        idx_mx = o_idx if o_idx < num_taps - 1 else num_taps - 1
        for f_idx in range(idx_mn, idx_mx):
            output[o_idx - group_delay] += taps[f_idx] * raw_iq[o_idx - f_idx]
    return output


def fractional_shift(
    tensor: np.ndarray, taps: np.ndarray, stride: int, delay: float
) -> np.ndarray:
    """Applies fractional sample delay of delay using a polyphase interpolator

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted in time.

        taps (:obj:`float` or :class:`numpy.ndarray`):
            taps to use for filtering

        stride (:obj:`int`):
            interpolation rate of internal filter

        delay (:obj:`float` ):
            Delay in number of samples in [-1, 1]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been fractionally-shifted along time dimension of size tensor.shape
    """
    real_part = _fractional_shift_helper(
        taps, tensor.real, stride, int(stride * float(delay)))
    imag_part = _fractional_shift_helper(
        taps, tensor.imag, stride, int(stride * float(delay)))
    tensor = real_part[: tensor.shape[0]] + 1j * imag_part[: tensor.shape[0]]
    zero_idx = -1 if delay < 0 else 0  # do not extrapolate, zero-pad.
    tensor[zero_idx] = 0
    return tensor


def amplitude_reversal(tensor: np.ndarray) -> np.ndarray:
    """Applies an amplitude reversal to the input tensor by multiplying by -1

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone an amplitude reversal

    """
    return tensor * -1


def amplitude_scale(tensor: np.ndarray, scale: float) -> np.ndarray:
    """Applies an amplitude scaling to the input tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        scale: (:obj:`float`):
            Scaling factor

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone an amplitude scaling

    """
    return tensor * scale


def roll_off(
    tensor: np.ndarray,
    lowercutfreq: float,
    uppercutfreq: float,
    num_taps: int,
) -> np.ndarray:
    """Applies front-end filter to tensor. Rolls off lower/upper edges of bandwidth

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        lowercutfreq (:obj:`float`):
            lower bandwidth cut-off to begin linear roll-off

        uppercutfreq (:obj:`float`):
            upper bandwidth cut-off to begin linear roll-off

        num_taps (:obj:`int`):
            order of each FIR filter to be applied

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone front-end filtering.

    """
    if (lowercutfreq == 0) & (uppercutfreq == 1):
        return tensor

    elif uppercutfreq == 1:
        if num_taps % 2 == 0:
            num_taps += 1
    bandwidth = uppercutfreq - lowercutfreq
    center_freq = lowercutfreq - 0.5 + bandwidth / 2
    sinusoid = np.exp(2j * np.pi * center_freq *
                      np.linspace(0, num_taps - 1, num_taps))
    taps = sp.firwin(
        num_taps,
        bandwidth,
        width=bandwidth * 0.02,
        window=sp.get_window("blackman", num_taps),
        scale=True,
    )
    taps = taps * sinusoid
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    return convolve_out[lidx:ridx]


def clip(tensor: np.ndarray, clip_percentage: float) -> np.ndarray:
    """Clips input tensor's values above/below a specified percentage of the
    max/min of the input tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        clip_percentage (:obj:`float`):
            Percentage of max/min values to clip

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_tensor = tensor.real
    max_val = np.max(real_tensor) * clip_percentage
    min_val = np.min(real_tensor) * clip_percentage
    real_tensor[real_tensor > max_val] = max_val
    real_tensor[real_tensor < min_val] = min_val

    imag_tensor = tensor.imag
    max_val = np.max(imag_tensor) * clip_percentage
    min_val = np.min(imag_tensor) * clip_percentage
    imag_tensor[imag_tensor > max_val] = max_val
    imag_tensor[imag_tensor < min_val] = min_val

    new_tensor = real_tensor + 1j * imag_tensor
    return new_tensor


def random_convolve(
    tensor: np.ndarray,
    num_taps: int,
    alpha: float,
) -> np.ndarray:
    """Create a complex-valued filter with `num_taps` number of taps, convolve
    the random filter with the input data, and sum the original data with the
    randomly-filtered data using an `alpha` weighting factor.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        num_taps: (:obj:`int`):
            Number of taps in random filter

        alpha: (:obj:`float`):
            Weighting for the summation between the original data and the
            randomly-filtered data, following:

            `output = (1 - alpha) * tensor + alpha * filtered_tensor`

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with weighted random filtering

    """
    filter_taps = np.random.rand(num_taps) + 1j * np.random.rand(num_taps)
    convolve_out = sp.convolve(tensor, filter_taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    return (1 - alpha) * tensor + alpha * convolve_out[lidx:ridx]


def drop_spec_samples(
    tensor: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str,
) -> np.ndarray:
    """Drop samples at specified input locations/durations with fill technique

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        drop_starts (:class:`numpy.ndarray`):
            Indices of where drops start

        drop_sizes (:class:`numpy.ndarray`):
            Durations of each drop instance

        fill (:obj:`str`):
            String specifying how the dropped samples should be replaced

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone the dropped samples

    """
    flat_spec = tensor.reshape(
        tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region_real = np.ones(
                drop_sizes[idx]) * flat_spec[0, drop_start - 1]
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start - 1]
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "bfill":
            drop_region_real = (
                np.ones(drop_sizes[idx]) * flat_spec[0,
                                                     drop_start + drop_sizes[idx]]
            )
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1,
                                                     drop_start + drop_sizes[idx]]
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "mean":
            drop_region_real = np.ones(drop_sizes[idx]) * np.mean(flat_spec[0])
            drop_region_complex = np.ones(
                drop_sizes[idx]) * np.mean(flat_spec[1])
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx])
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        elif fill == "min":
            drop_region_real = np.ones(
                drop_sizes[idx]) * np.min(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.min(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "max":
            drop_region_real = np.ones(
                drop_sizes[idx]) * np.max(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.max(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "low":
            drop_region = np.ones(drop_sizes[idx]) * 1e-3
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        elif fill == "ones":
            drop_region = np.ones(drop_sizes[idx])
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        else:
            raise ValueError(
                "fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {}".format(
                    fill
                )
            )
    new_tensor = flat_spec.reshape(
        tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return new_tensor


def spec_patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        patch_size (:obj:`int`):
            Size of each patch to shuffle

        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    channels, height, width = tensor.shape
    num_freq_patches = int(height / patch_size)
    num_time_patches = int(width / patch_size)
    num_patches = int(num_freq_patches * num_time_patches)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        freq_idx = np.floor(patch_idx / num_freq_patches)
        time_idx = patch_idx % num_time_patches
        patch = tensor[
            :,
            int(freq_idx * patch_size): int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size): int(time_idx * patch_size + patch_size),
        ]
        patch = patch.reshape(int(2 * patch_size * patch_size))
        np.random.shuffle(patch)
        patch = patch.reshape(2, int(patch_size), int(patch_size))
        tensor[
            :,
            int(freq_idx * patch_size): int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size): int(time_idx * patch_size + patch_size),
        ] = patch
    return tensor


def spec_translate(
    tensor: np.ndarray,
    time_shift: int,
    freq_shift: int,
) -> np.ndarray:
    """Apply time/freq translation to input spectrogram

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        time_shift (:obj:`int`):
            Time shift

        freq_shift (:obj:`int`):
            Frequency shift

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone time/freq translation

    """
    # Pre-fill the data with background noise
    new_tensor = np.random.rand(*tensor.shape) * \
        np.percentile(np.abs(tensor), 50)

    # Apply translation
    channels, height, width = tensor.shape
    if time_shift >= 0 and freq_shift >= 0:
        valid_dur = width - time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:,
                   time_shift:] = tensor[:, :valid_bw, :valid_dur]
    elif time_shift < 0 and freq_shift >= 0:
        valid_dur = width + time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:, :valid_dur] = tensor[:,
                                                        :valid_bw, -time_shift:]
    elif time_shift >= 0 and freq_shift < 0:
        valid_dur = width - time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, time_shift:] = tensor[:, -
                                                       freq_shift:, :valid_dur]
    elif time_shift < 0 and freq_shift < 0:
        valid_dur = width + time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, :valid_dur] = tensor[:, -
                                                      freq_shift:, -time_shift:]

    return new_tensor


def spectrogram_image(
    tensor: np.ndarray,
    black_hot: bool = True
) -> np.ndarray:
    """tensor to image

    Args:
        tensor (numpy.ndarray): (batch_size, vector_length, ...)-sized tensor
        black_hot (bool, optional): toggles black hot spectrogram. Defaults to False (white-hot).


    Returns:
        Tensor:
            array.
    """
    spec = 10 * np.log10(tensor+np.finfo(np.float32).tiny)
    img = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    if black_hot:
        img = cv2.bitwise_not(img, img)

    return img
