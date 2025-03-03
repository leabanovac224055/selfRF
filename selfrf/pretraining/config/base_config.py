from dataclasses import dataclass, fields
import argparse
from typing import Optional

import torch

from selfrf.pretraining.utils.enums import BackboneProvider, BackboneType, DatasetType

# Default values as constants
DEFAULT_DATASET = DatasetType.TORCHSIG_NARROWBAND
DEFAULT_ROOT = './datasets'
DEFAULT_DOWNLOAD = False
DEFAULT_FAMILY = False
DEFAULT_SPECTROGRAM = False
DEFAULT_NFFT = 512
DEFAULT_NOVERLAP = 0
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4

DEFAULT_BACKBONE = BackboneType.RESNET50
DEFAULT_BACKBONE_PROVIDER = BackboneProvider.TIMM

DEFAULT_EMBEDDING_DIM = 2048
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TO_FLOAT_32 = False
DATASET_IQ_SAMPLES = {
    DatasetType.TORCHSIG_WIDEBAND: 262144,
    DatasetType.TORCHSIG_NARROWBAND: 4096,
}


@dataclass
class BaseConfig:
    dataset: DatasetType = DEFAULT_DATASET
    dataset_name: str = None
    root: str = DEFAULT_ROOT
    download: bool = False
    family: bool = DEFAULT_FAMILY
    impairment_level: int = 2

    # Add private storage field
    _custom_iq_samples: Optional[int] = None

    @property
    def num_iq_samples(self) -> int:
        """Get number of IQ samples based on dataset type or user override."""
        if self._custom_iq_samples is not None:
            return self._custom_iq_samples
        return DATASET_IQ_SAMPLES.get(self.dataset)

    @num_iq_samples.setter
    def num_iq_samples(self, value: int):
        """Set custom number of IQ samples."""
        self._custom_iq_samples = value

    spectrogram: bool = DEFAULT_SPECTROGRAM
    nfft: int = DEFAULT_NFFT
    noverlap: int = DEFAULT_NOVERLAP

    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS

    backbone: BackboneType = DEFAULT_BACKBONE
    backbone_provider: BackboneProvider = DEFAULT_BACKBONE_PROVIDER
    embedding_dim: int = DEFAULT_EMBEDDING_DIM

    device: torch.device = DEFAULT_DEVICE
    to_float_32: bool = DEFAULT_TO_FLOAT_32


def add_base_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--dataset',
        type=lambda x: DatasetType(x),
        choices=list(DatasetType),
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--root',
        type=str,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        '--download',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_DOWNLOAD,
    )
    parser.add_argument(
        '--family',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_FAMILY,
    )
    parser.add_argument(
        '--impairment-level',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--num-iq-samples',
        type=int,
        default=None,
        help=(f'Number of IQ samples to use. Defaults based on dataset: '
              f'Wideband={DATASET_IQ_SAMPLES[DatasetType.TORCHSIG_WIDEBAND]}, '
              f'Narrowband={DATASET_IQ_SAMPLES[DatasetType.TORCHSIG_NARROWBAND]}. '
              f'Set this to override the default for your dataset.')
    )
    parser.add_argument(
        '--spectrogram',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_SPECTROGRAM,
    )
    parser.add_argument(
        '--nfft',
        type=int,
        default=DEFAULT_NFFT,
    )
    parser.add_argument(
        '--noverlap',
        type=int,
        default=DEFAULT_NOVERLAP,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
    )
    parser.add_argument(
        '--backbone',
        type=BackboneType.from_string,
        choices=list(BackboneType),
        default=DEFAULT_BACKBONE,
    )
    parser.add_argument(
        '--backbone-provider',
        type=BackboneProvider.from_string,
        choices=list(BackboneProvider),
        default=DEFAULT_BACKBONE_PROVIDER,
        help='Provider library for the backbone implementation'
    )
    parser.add_argument(
        '--device',
        type=lambda x: torch.device(x),
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        '--to-float-32',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_TO_FLOAT_32,
    )


def parse_base_config(parser: argparse.ArgumentParser) -> BaseConfig:
    """Parse command line arguments into a BaseConfig object."""
    args = parser.parse_args()
    args_dict = vars(args)

    # Get all field names from BaseConfig
    base_field_names = {f.name for f in fields(BaseConfig)}

    # Handle num_iq_samples specially
    custom_samples = args_dict.pop('num_iq_samples', None)

    # Filter args_dict to only include fields in BaseConfig
    filtered_args = {k: v for k,
                     v in args_dict.items() if k in base_field_names}

    # Create config with only the fields BaseConfig knows about
    config = BaseConfig(**filtered_args)

    # Set custom samples if provided
    if custom_samples is not None:
        config.num_iq_samples = custom_samples

    return config


def print_config(config: BaseConfig) -> None:
    """Print config in a structured format"""
    print("\nConfiguration:")

    # Get fields directly from config instance
    for field in fields(config):
        # Skip private fields that start with underscore
        if field.name.startswith('_'):
            continue

        if field.name == 'num_iq_samples':
            # Get property value for num_iq_samples
            value = config.num_iq_samples
        else:
            value = getattr(config, field.name)
        print(f"  {field.name}: {value}")
