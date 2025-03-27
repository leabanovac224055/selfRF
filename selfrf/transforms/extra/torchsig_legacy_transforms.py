"""TorchSig legacy Transforms from v0.6.0
"""


from typing import List, Tuple, Union
import numpy as np

from torchsig.signals.signal_types import DatasetSignal
from torchsig.transforms.dataset_transforms import DatasetTransform
from torchsig.utils.dsp import torchsig_complex_data_type
import torchsig.transforms.functional as torchsig_F

import selfrf.transforms.extra.torchsig_legacy_functional as F_LEGACY

__all__ = [
    "Identity",
    "RandomTimeShift",
    "AmplitudeReversal",
    "RandomFrequencyShift",
    "RandomPhaseShift",
    "TargetSNR",
]


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
        shift_range: Union[List, Tuple] = (-10, 10),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_shift = shift_range[0]
        self.max_shift = shift_range[1]

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        shift = np.random.uniform(self.min_shift, self.max_shift)

        signal.data = np.ascontiguousarray(
            F_LEGACY.time_shift(signal.data, int(shift)))

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


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

    def __init__(
        self,
        freq_shift_range: Union[list, tuple] = (-0.3, 0.3),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Store range directly
        self.min_freq = freq_shift_range[0]
        self.max_freq = freq_shift_range[1]

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply frequency shift to signal."""
        freq_shift = np.random.uniform(self.min_freq, self.max_freq)

        # Apply frequency shift
        signal.data = np.ascontiguousarray(
            F_LEGACY.freq_shift(signal.data, freq_shift))

        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class RandomPhaseShift(DatasetTransform):
    """Applies a random phase shift to the signal.

    A phase shift rotates the signal in the complex plane by multiplying 
    by e^(jθ), where θ is the phase shift in radians.

    Args:
        phase_range (:obj:`tuple`):
            Range of phase shift in radians (min, max)
    """

    def __init__(
        self,
        phase_range: Tuple[float, float] = (0, np.pi/4),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.phase_range_distribution = self.get_distribution(phase_range)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply phase shift to signal."""
        # Get random phase shift value
        phase_shift = self.phase_range_distribution()

        # Create phase shift factor e^(jθ)
        phase_factor = np.exp(1j * phase_shift)

        # Apply the phase shift by multiplication
        signal.data = signal.data * phase_factor

        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class TargetSNR(DatasetTransform):
    """Sets a target SNR by adding appropriate white Gaussian noise.

    This simplified version ignores signal metadata and assumes the input
    is entirely the signal of interest.

    Args:
        target_snr_range (:obj:`tuple`):
            Target SNR in dB, specified as (min, max) range
        linear (:obj:`bool`):
            If True, SNR values are in linear scale, not dB. Default is False.
    """

    def __init__(
        self,
        target_snr_range: Tuple[float, float] = (0, 30),
        linear: bool = False,
        debug: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.linear = linear
        self.debug = debug
        self.min_snr = target_snr_range[0]
        self.max_snr = target_snr_range[1]

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Apply noise to achieve target SNR."""
        # Get target SNR from distribution
        target_snr_db = np.random.uniform(self.min_snr, self.max_snr)

        # Convert from linear if needed
        if self.linear:
            target_snr_db = 10 * np.log10(target_snr_db)

        # Store original signal for SNR verification
        original_signal = signal.data.copy()

        # Calculate current signal power in dB
        signal_power = np.mean(np.abs(signal.data)**2)
        signal_power_db = 10 * np.log10(signal_power)

        # Calculate required noise power to achieve target SNR
        noise_power_db = signal_power_db - target_snr_db

        # Add noise using torchsig function
        signal.data = torchsig_F.awgn(
            signal.data,
            noise_power_db=noise_power_db,
        )

        # Calculate actual SNR achieved
        if self.debug:
            # Calculate added noise by subtracting original signal
            noise = signal.data - original_signal
            noise_power = np.mean(np.abs(noise)**2)

            # Calculate actual SNR in dB
            actual_snr_db = 10 * \
                np.log10(signal_power /
                         noise_power) if noise_power > 0 else float('inf')

            # Print debugging information
            print(
                f"TargetSNR: Target={target_snr_db:.2f}dB, Actual={actual_snr_db:.2f}dB, Difference={actual_snr_db-target_snr_db:.2f}dB")

        # Ensure data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # Update any relevant signal metadata
        self.update(signal)
        return signal
