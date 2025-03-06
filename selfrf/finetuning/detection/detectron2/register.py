
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.wideband import StaticWideband
from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.transforms.dataset_transforms import Spectrogram
from torchsig.transforms.target_transforms import (
    ClassName,
    FamilyName,
    ClassIndex,
    FamilyIndex,
)

from selfrf.transforms import (
    SpectrogramNormalize,
)
from selfrf.transforms.extra.target_transforms import BBOXLabel

from .create_coco import convert_datamodule_to_coco

SEED = 123456789
FFT_SIZE = 512
NUM_SAMPLES = 1000


def register_dataset(
    root: Path,
    dataset_path: str,
    download: bool = False,
    force: bool = False,
):
    """Register RF COCO format dataset with detectron2"""

    datamodule = WidebandDataModule(
        root=root,
        dataset_metadata=WidebandMetadata(
            seed=SEED,
            num_iq_samples_dataset=FFT_SIZE * FFT_SIZE,
            impairment_level=2,
            fft_size=FFT_SIZE,
            num_signals_min=1,
            num_signals_max=3,
        ),
        num_samples_train=NUM_SAMPLES,
        transforms=[Spectrogram(
            fft_size=FFT_SIZE,
        )],
        target_transforms=[
            BBOXLabel(),  # bbox
            FamilyName(),  # category name
            FamilyIndex(),  # category index
            FamilyName(),  # super category name
        ],
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    path_to_coco = convert_datamodule_to_coco(datamodule, dataset_path, force)

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


def to_detectron2_dicts(dataset: StaticWideband):
    """Convert dataset to detectron2 dicts"""
    print(dataset.dataset_metadata)
    # load labels
