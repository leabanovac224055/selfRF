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
    """Factory for Using Pre-Extracted IQDM Datasets"""

    _iqdm_dataset_registry: Dict[DatasetType, str] = {
        DatasetType.IQDM_NARROWBAND: "datasets/iqdm_narrowband",
        DatasetType.IQDM_WIDEBAND: "datasets/iqdm_wideband"
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> TorchSigDataModule:
        """Create a dataset using pre-extracted IQDM datasets."""
        dataset_type = DatasetType(config.dataset)

        if dataset_type not in cls._iqdm_dataset_registry:
            raise ValueError(f"❌ {dataset_type} is not a valid IQDM dataset.")

        dataset_root = cls._iqdm_dataset_registry[dataset_type]
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"❌ Dataset path {dataset_root} does not exist! Run `extract_rf_bands.py` first."
            )

        dataset_metadata = get_dataset_metadata(config)

        dataset_class = NarrowbandDataModule if dataset_type == DatasetType.IQDM_NARROWBAND else WidebandDataModule

        import json

        import json

        class ExtractClassIndexTransform:
            """TorchSig-compatible class to extract class_index from metadata."""

            def __init__(self):
                # This attribute is required by TorchSig
                self.targets_metadata = ["class_index"]

            def __call__(self, metadata):
                print(
                    f"🔍 Metadata Received (before processing): {metadata} (Type: {type(metadata)})")

                # If metadata is a list, take the first element
                if isinstance(metadata, list):
                    if len(metadata) > 0:
                        metadata = metadata[0]
                    else:
                        metadata = {}

                # If metadata is a string, try to decode it from JSON
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        print(
                            f"⚠️ Failed to parse metadata string: {metadata}, defaulting to empty dict")
                        metadata = {}

                # If metadata is not a dict, default to an empty dict
                if not isinstance(metadata, dict):
                    print(
                        f"⚠️ Unexpected metadata format: {type(metadata)}, defaulting to empty dict")
                    metadata = {}

                # Extract class_index, defaulting to 0
                class_index = metadata.get("class_index", 0)
                try:
                    class_index = int(class_index)
                except ValueError:
                    print(
                        f"⚠️ Invalid class_index: {class_index}, defaulting to 0")
                    class_index = 0

                print(f"✅ Extracted class_index: {class_index}")
                # Return a list containing the dictionary—this is what TorchSig expects.
                return [{"class_index": class_index}]

        return dataset_class(
            root=dataset_root,
            dataset_metadata=dataset_metadata,
            num_samples_train=100,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=build_transform(config),
            target_transforms=[ExtractClassIndexTransform()],
            collate_fn=build_collate_fn(config),
        )


def get_dataset_metadata(config: BaseConfig) -> DatasetMetadata:
    if config.dataset == DatasetType.TORCHSIG_NARROWBAND:
        return get_narrowband_metadata(config)
    elif config.dataset == DatasetType.TORCHSIG_WIDEBAND:
        return get_wideband_metadata(config)
    elif config.dataset == DatasetType.IQDM_NARROWBAND:
        return get_iqdm_narrowband_metadata(config)
    elif config.dataset == DatasetType.IQDM_WIDEBAND:
        return get_iqdm_wideband_metadata(config)
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


def get_iqdm_narrowband_metadata(config: BaseConfig) -> NarrowbandMetadata:
    """Returns metadata for IQDM Narrowband datasets."""
    return NarrowbandMetadata(
        num_iq_samples_dataset=config.num_iq_samples,
        impairment_level=config.impairment_level,
        fft_size=config.nfft,
    )


def get_iqdm_wideband_metadata(config: BaseConfig) -> WidebandMetadata:
    """Returns metadata for IQDM Wideband datasets."""
    return WidebandMetadata(
        num_iq_samples_dataset=config.num_iq_samples,
        impairment_level=config.impairment_level,
        fft_size=config.nfft,
        num_signals_min=1,
        num_signals_max=5,
    )


def build_dataloader(config: BaseConfig, use_static: bool = False) -> TorchSigDataModule:
    """
    Build and return the TorchSig dataloader.

    Args:
        config (BaseConfig): Configuration object
        use_static (bool): If True, load from `iqdm_*` datasets; otherwise, use TorchSig generation.
    """
    if use_static:
        return IQDMStaticDatasetFactory.create_dataset(config)
    return DatasetFactory.create_dataset(config)
