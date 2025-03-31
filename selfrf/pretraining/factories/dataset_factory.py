import os
import torch
import json
from typing import Dict, Type
from torchsig.datasets.datamodules import TorchSigDataModule, NarrowbandDataModule, WidebandDataModule
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform
from selfrf.transforms.extra.target_transforms import ConstantTargetTransform
from selfrf.data.sigmf.iqdm_modules import IQDMNarrowbandDataModule, IQDMWidebandDataModule


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
            num_samples_train=100,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=build_transform(config),
            target_transforms=build_target_transform(config),
            collate_fn=build_collate_fn(config),
        )


class IQDMStaticDatasetFactory:
    _dataset_registry: Dict[DatasetType, Type[TorchSigDataModule]] = {
        DatasetType.IQDM_NARROWBAND: IQDMNarrowbandDataModule,
        DatasetType.IQDM_WIDEBAND: IQDMWidebandDataModule
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> TorchSigDataModule:
        dataset_class = cls._dataset_registry[config.dataset]
        return dataset_class(
            config=config,
            root=config.root,
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
    return NarrowbandMetadata(
        num_iq_samples_dataset=config.num_iq_samples,
        impairment_level=config.impairment_level,
        fft_size=config.nfft,
    )


def get_wideband_metadata(config: BaseConfig) -> WidebandMetadata:
    return WidebandMetadata(
        num_iq_samples_dataset=config.num_iq_samples,
        impairment_level=config.impairment_level,
        fft_size=config.nfft,
        num_signals_min=1,
        num_signals_max=5,
    )


def build_dataloader(config: BaseConfig, use_static: bool = True) -> TorchSigDataModule:
    """
    Build and return the TorchSig dataloader.

    Args:
        config (BaseConfig): Configuration object
        use_static (bool): If True, load from `iqdm_*` datasets; otherwise, use TorchSig generation.
    """
    if use_static:
        return IQDMStaticDatasetFactory.create_dataset(config)
    return DatasetFactory.create_dataset(config)
