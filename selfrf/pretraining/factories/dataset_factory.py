from typing import Dict, Type

from torchsig.datasets.datamodules import TorchSigDataModule, NarrowbandDataModule, WidebandDataModule
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform


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
            # dataset_name=config.dataset_name,
            # download=config.download,
            dataset_metadata=NarrowbandMetadata(
                num_iq_samples_dataset=config.num_iq_samples,
                impairment_level=2,
                fft_size=config.nfft,
            ),
            num_samples_train=100,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=build_transform(config),
            target_transforms=build_target_transform(config),
            collate_fn=build_collate_fn(config),
        )


def build_dataloader(config: BaseConfig) -> TorchSigDataModule:
    return DatasetFactory.create_dataset(config)
