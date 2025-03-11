from typing import Dict, Type

from torchsig.datasets.datamodules import TorchSigDataModule, NarrowbandDataModule, WidebandDataModule
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.default_configs.loader import get_default_yaml_config
from torchsig.datasets.dataset_utils import to_dataset_metadata

from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform

FFT_SIZE = 512


class DatasetFactory:
    _dataset_registry: Dict[DatasetType, Type[TorchSigDataModule]] = {
        DatasetType.TORCHSIG_NARROWBAND: NarrowbandDataModule,
        DatasetType.TORCHSIG_WIDEBAND: WidebandDataModule
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> TorchSigDataModule:
        """Create dataset from config"""
        dataset_type = DatasetType(config.dataset)
        dataset_class = cls._dataset_registry[dataset_type]

        return dataset_class(
            root=config.root,
            dataset_metadata=get_dataset_metadata(config),
            num_samples_train=config.num_samples,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=build_transform(config),
            target_transforms=build_target_transform(config),
            collate_fn=build_collate_fn(config),
        )


def get_dataset_metadata(config: BaseConfig) -> DatasetMetadata:
    if config.dataset == DatasetType.TORCHSIG_NARROWBAND:
        return get_narrowband_metadata(config)
    elif config.dataset == DatasetType.TORCHSIG_WIDEBAND:
        return get_wideband_metadata(config)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")


def get_narrowband_metadata(config: BaseConfig) -> NarrowbandMetadata:
    metadata = get_default_yaml_config(
        dataset_type="narrowband",
        impairment_level=2,
        train=True,
    )
    metadata["overrides"]["snr_db_min"] = 10
    metadata["overrides"]["signal_bandwidth_min"] = 1_000_000
    metadata["overrides"]["signal_bandwidth_max"] = 1_000_0000
    metadata["overrides"]["impairment_level"] = 2
    metadata["overrides"]["num_iq_samples_dataset"] = FFT_SIZE**2
    metadata["overrides"]["fft_size"] = FFT_SIZE

    # Set valid duration bounds based on constraints
    max_duration = 0.00262144  # max allowed for FFT_SIZE=512
    min_duration = 0.00131072  # min required based on error message

    metadata["overrides"]["signal_duration_max"] = max_duration
    metadata["overrides"]["signal_duration_min"] = min_duration
    metadata = to_dataset_metadata(metadata)
    return metadata


def get_wideband_metadata(config: BaseConfig) -> WidebandMetadata:
    metadata = get_default_yaml_config(
        dataset_type="wideband",
        impairment_level=2,
        train=True,
    )
    metadata["overrides"]["snr_db_min"] = 10
    metadata["overrides"]["signal_bandwidth_min"] = 1_000_000
    metadata["overrides"]["signal_bandwidth_max"] = 1_000_0000
    metadata["overrides"]["impairment_level"] = 2
    metadata["overrides"]["num_iq_samples_dataset"] = FFT_SIZE**2
    metadata["overrides"]["fft_size"] = FFT_SIZE

    # Set valid duration bounds based on constraints
    max_duration = 0.00262144  # max allowed for FFT_SIZE=512
    min_duration = 0.00131072  # min required based on error message

    metadata["overrides"]["signal_duration_max"] = max_duration
    metadata["overrides"]["signal_duration_min"] = min_duration
    metadata = to_dataset_metadata(metadata)
    return metadata


def build_dataloader(config: BaseConfig) -> TorchSigDataModule:
    return DatasetFactory.create_dataset(config)
