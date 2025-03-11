
import json
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


import numpy as np
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.wideband import StaticWideband
from torchsig.datasets.default_configs.loader import get_default_yaml_config
from torchsig.datasets.dataset_utils import to_dataset_metadata
from torchsig.transforms.dataset_transforms import Spectrogram
from torchsig.transforms.base_transforms import Compose

from torchsig.transforms.target_transforms import (
    ClassName,
    FamilyName,
    ClassIndex,
    FamilyIndex,
    SNR,
)

from selfrf.finetuning.detection.detectron2.config import Detectron2Config
from selfrf.transforms import (
    SpectrogramImage,
)
from selfrf.transforms.extra.target_transforms import BBOXLabel, ConstantFamilyName, ConstantSignalIndex, ConstantSignalName

from .create_coco import convert_datamodule_to_coco

FFT_SIZE = 512
NUM_SAMPLES = 100


def register_dataset(
    config: Detectron2Config,
):
    """Register RF COCO format dataset with detectron2"""
    root = Path(config.root)
    dataset_path = Path(config.dataset_path)

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

    print(json.dumps(metadata, indent=4))
    metadata = to_dataset_metadata(metadata)

    datamodule = WidebandDataModule(
        root=root / dataset_path,
        dataset_metadata=metadata,
        num_samples_train=NUM_SAMPLES,
        transforms=[
            Compose([
                Spectrogram(
                    fft_size=FFT_SIZE,
                ),
                SpectrogramImage()
            ]),
        ],
        target_transforms=get_target_transforms(config=config),
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    path_to_coco = convert_datamodule_to_coco(
        datamodule, config.force_recreation)

    dataset_name = "torchsig_wideband"

    # Register datasets
    register_coco_instances(
        f"{dataset_name}_train",
        {},
        build_annotations_path(path_to_coco, "train"),
        build_images_path(path_to_coco, "train"),
    )
    register_coco_instances(
        f"{dataset_name}_val",
        {},
        build_annotations_path(path_to_coco, "val"),
        build_images_path(path_to_coco, "val")
    )

    print("Dataset registered successfully!")


def build_images_path(coco_path: Path, split: str) -> str:
    """Build path to images"""
    return str(coco_path / "images" / split)


def build_annotations_path(coco_path: Path, split: str) -> str:
    """Build path to annotations"""
    return str(coco_path / "annotations" / f"instances_{split}.json")


def get_target_transforms(
    config: Detectron2Config,
) -> list:
    """Get target transform for detectron2"""

    if config.mode == "detection":
        return [
            BBOXLabel(),  # bbox
            ConstantSignalName("signal"),  # category name
            ConstantSignalIndex(0),  # category index
            ConstantFamilyName("signal"),  # super category name
            SNR(),  # SNR
        ]
    elif config.mode == "recognition":
        return [
            BBOXLabel(),  # bbox
            ClassName(),  # category name
            ClassIndex(),  # category index
            FamilyName(),  # super category name
            SNR(),  # SNR
        ]
    elif config.mode == "family_recognition":
        return [
            BBOXLabel(),  # bbox
            FamilyName(),  # category name
            FamilyIndex(),  # category index
            FamilyName(),  # super category name
            SNR(),  # SNR
        ]
