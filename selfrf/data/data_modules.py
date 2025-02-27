from functools import partial
import json
import os
from typing import Callable, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from dotenv import load_dotenv
import lightning.pytorch as pl
from minio import Minio
from tqdm import tqdm

from torch.utils.data import DataLoader

from torchsig.transforms.base_transforms import Transform
from selfrf.transforms.extra.torchsig_legacy_transforms import Identity

from selfrf.data.datasets import RFCOCODataset


class RFCOCODataModule(pl.LightningDataModule):
    """Base DataModule for RFCOCO datasets"""

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None,
        collate_fn: Callable = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform or Identity()
        self.target_transform = target_transform or Identity()
        self.collate_fn = collate_fn

        self.train_dataset = None  # Dataset: train dataset
        self.val_dataset = None  # Dataset: validation dataset

        self.data_path = None  # str: Path to downloaded dataset in root
        self.train_path = None  # str: Path to train dataset
        self.val_path = None  # str: Path to validation dataset

    def setup(self, stage: Optional[str] = None):
        """Set up datasets, self.train and self.val

        Args:
            stage (str): PyTorch Lightning trainer stage - fit, test, predict.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("setup not implemented")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class TorchsigNarrowbandRFCOCODataModule(RFCOCODataModule):

    def __init__(self,
                 dataset_name: str = "narrowband_impaired",
                 download: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv()

        self.dataset_name = dataset_name
        self.bucket = "iqdm-ai"

        self.download = download
        if self.download:
            self.minio = Minio(
                endpoint=os.getenv("MINIO_ENDPOINT"),
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
                cert_check=os.getenv("MINIO_CERT_CHECK",
                                     "true").lower() == "true",
            )

    def prepare_data(self):
        """Download RF COCO dataset from Minio"""

        if self.download:
            _download(
                root=self.root,
                bucket=self.bucket,
                dataset_name=self.dataset_name,
                minio=self.minio,)
        else:
            raise NotImplementedError(
                "prepare_data not implemented for ahoc dataset generation")

    def setup(self, stage=None):

        if stage == "fit":

            self.train_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )

            self.val_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            raise NotImplementedError(
                "setup not implemented for stage: {stage}")


class TorchsigWidebandRFCOCODataModule(RFCOCODataModule):
    def __init__(self,
                 dataset_name: str = "wideband_impaired",
                 download: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv()

        self.dataset_name = dataset_name
        self.bucket = "iqdm-ai"
        self.download = download

        if self.download:
            self.minio = Minio(
                endpoint=os.getenv("MINIO_ENDPOINT"),
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
                cert_check=os.getenv("MINIO_CERT_CHECK",
                                     "true").lower() == "true",
            )

    def prepare_data(self):
        """Download RF COCO dataset from Minio"""

        print(f"Checking if dataset exists at {self.root / self.dataset_name}")
        # check if dataset is already exists
        if _check_if_dataset_exists(self.root, self.dataset_name):
            print(
                f"Dataset already exists at {self.root / self.dataset_name}. Skipping download or generation.")
            return

        if self.download:
            _download(
                root=self.root,
                bucket=self.bucket,
                dataset_name=self.dataset_name,
                minio=self.minio,)
        else:
            raise NotImplementedError(
                "prepare_data not implemented for ahoc dataset generation")

    def setup(self, stage: str = None):

        if stage == "fit":

            self.train_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )

            self.val_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )

        else:
            raise NotImplementedError(
                "setup not implemented for stage: {stage}")


def _download(
        root: Path,
        bucket: str,
        dataset_name: str,
        minio: Minio,
        max_workers: int = 10,
):
    download_path = root / dataset_name
    print(f"Downloading dataset to {download_path}")
    download_path.mkdir(parents=True, exist_ok=True)

    # Download annotation files
    for split in ["train", "val"]:
        annot_file = f"instances_{split}.json"
        annot_file_local_path = download_path / "annotations" / annot_file
        annot_file_local_path.parent.mkdir(parents=True, exist_ok=True)

        minio.fget_object(
            bucket,
            f"{dataset_name}/annotations/{annot_file}",
            str(annot_file_local_path)
        )

        annotations = json.loads(annot_file_local_path.read_text())

        # Download IQ files in parallel
        download_fn = partial(_download_iq_frame,
                              root=root,
                              dataset_name=dataset_name,
                              bucket=bucket,
                              split=split,
                              minio=minio)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_fn, frame)
                       for frame in annotations["iq_frames"]]

            list(tqdm(
                concurrent.futures.as_completed(futures),
                total=len(annotations["iq_frames"]),
                desc=f"Downloading {split} IQ frames"
            ))


def _download_iq_frame(
    frame: Dict,
    split: str,
    root: Path,
    dataset_name: Path,
    bucket: str,
    minio: Minio,
):
    """Helper function to download a single IQ frame"""
    download_path = root / dataset_name
    iq_local_path = download_path / split / str(frame["file_name"])
    iq_local_path.parent.mkdir(parents=True, exist_ok=True)

    minio.fget_object(
        bucket,
        f"{dataset_name}/{split}/{frame['file_name']}",
        str(iq_local_path)
    )
    return frame["file_name"]


def _check_if_dataset_exists(root: Path, dataset_name: str):

    for split in ["train", "val"]:
        annot_file = f"instances_{split}.json"
        annot_file_local_path = root / dataset_name / "annotations" / annot_file

        if not annot_file_local_path.exists():
            return False

    return True
