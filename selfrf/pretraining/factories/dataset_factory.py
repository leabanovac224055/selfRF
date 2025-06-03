from typing import Dict, Type

from torchsig.datasets.datamodules import TorchSigDataModule, NarrowbandDataModule, WidebandDataModule
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.default_configs.loader import get_default_yaml_config
from torchsig.datasets.dataset_utils import to_dataset_metadata
from torchsig.signals.signal_lists import TorchSigSignalLists

from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform
from selfrf.data.iqdm.iqdm_modules import IQDMNarrowbandDataModule, IQDMWidebandDataModule
from selfrf.data.meta.two_tower_dataset import TwoTowerDataModule


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
            transforms=[build_transform(config)],
            target_transforms=build_target_transform(config),
            collate_fn=build_collate_fn(config),
        )


class IQDMStaticDatasetFactory:
    _dataset_registry: Dict[DatasetType, Type[TorchSigDataModule]] = {
        DatasetType.IQDM_NARROWBAND: IQDMNarrowbandDataModule,
        DatasetType.IQDM_WIDEBAND: IQDMWidebandDataModule,
        DatasetType.TWO_TOWER_NARROWBAND: TwoTowerDataModule,
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> TorchSigDataModule:
        dataset_class = cls._dataset_registry[config.dataset]
        kwargs = dict(
            config=config,
            root=config.root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=build_transform(config),
            target_transforms=build_target_transform(config),
        )

        return dataset_class(**kwargs)

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

    metadata["overrides"]["snr_db_min"] = 20
    metadata["overrides"]["sample_rate"] = 10_000_000
    metadata["overrides"]["signal_bandwidth_min"] = 1_500_000
    metadata["overrides"]["signal_bandwidth_max"] = 2_000_000
    metadata["overrides"]["num_iq_samples_dataset"] = 4096
    metadata["overrides"]["fft_size"] = config.nfft

    metadata = to_dataset_metadata(metadata)
    return metadata


def get_wideband_metadata(config: BaseConfig) -> WidebandMetadata:
    metadata = get_default_yaml_config(
        dataset_type="wideband",
        impairment_level=2,
        train=True,
    )
    metadata["overrides"]["snr_db_min"] = 10
    metadata["overrides"]["signal_bandwidth_min"] = 5_000_000
    metadata["overrides"]["signal_bandwidth_max"] = 13_000_000
    metadata["overrides"]["impairment_level"] = 2
    metadata["overrides"]["num_iq_samples_dataset"] = config.nfft**2
    metadata["overrides"]["fft_size"] = config.nfft

    # Set valid duration bounds based on constraints
    max_duration = 0.00065536  # Maximum allowed value per error message
    min_duration = 0.00016384  # Minimum required value per error message

    metadata["overrides"]["signal_duration_max"] = max_duration
    metadata["overrides"]["signal_duration_min"] = min_duration
    metadata = to_dataset_metadata(metadata)
    return metadata


def build_dataloader(config: BaseConfig) -> TorchSigDataModule:
    """Build the dataloader based on the dataset type."""
    if config.dataset in {DatasetType.IQDM_NARROWBAND, DatasetType.IQDM_WIDEBAND, DatasetType.TWO_TOWER_NARROWBAND}:
        return IQDMStaticDatasetFactory.create_dataset(config)
    else:
        return DatasetFactory.create_dataset(config)
