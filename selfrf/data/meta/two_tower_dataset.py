import os
import zarr
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from selfrf.pretraining.utils import Signal  # your custom Signal class
from selfrf.pretraining.config import BaseConfig


class TorchSigMetadata:
    """Wraps a metadata dictionary into an attribute-accessible object."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.applied_transforms = []  # ✅ Required for TorchSig transform tracking


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        feature_vector_path: str,
        transform=None,
        target_transform=None,
    ):
        # Open IQ data (.zarr)
        self.zarr_data = zarr.open_array(zarr_path, mode='r')
        self.attrs = self.zarr_data.attrs.asdict()

        self.transform = transform
        self.target_transform = target_transform

        # Load metadata feature vectors from JSON
        with open(feature_vector_path, 'r') as f:
            self.feature_vectors = json.load(f)

        assert len(self.zarr_data) == len(self.feature_vectors), (
            f"Mismatch between IQ samples ({len(self.zarr_data)}) and "
            f"metadata vectors ({len(self.feature_vectors)})"
        )

    def __len__(self):
        return len(self.zarr_data)

    def __getitem__(self, idx):
        # ----- Load IQ signal -----
        iq = self.zarr_data[idx]
        raw_metadata = self.attrs.get(str(idx), {})

        if isinstance(raw_metadata, list) and len(raw_metadata) == 1 and isinstance(raw_metadata[0], dict):
            raw_metadata = raw_metadata[0]

        metadata_obj = TorchSigMetadata(raw_metadata)

        # Wrap IQ + metadata in Signal object
        signal = Signal(data=iq, metadata=[metadata_obj])

        # Transform to get views
        if self.transform is not None:
            transformed = self.transform(signal)
            view1, view2 = transformed.data
        else:
            view1 = view2 = signal

        # ----- Load metadata vector -----
        vector_dict = self.feature_vectors[idx]
        metadata_vector = torch.tensor(
            [vector_dict[k] for k in vector_dict if k != "class_index"], dtype=torch.float32
        )

        label = int(raw_metadata.get("class_index", 0))

        return ((view1, view2), metadata_vector, label)


class TwoTowerDataModule(LightningDataModule):
    def __init__(
        self,
        config: BaseConfig,
        root: str,
        batch_size: int,
        num_workers: int,
        transforms,
        target_transforms,
        collate_fn,
    ):
        self.config = config
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.collate_fn = collate_fn

        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = False
        self.dataset = None

    def setup(self, stage=None):
        zarr_path = f"{self.root}/NARROWBAND_ZARR/data.zarr"
        feature_path = f"{self.root}/NARROWBAND_ZARR/feature_vectors.json"

        self.dataset = TwoTowerDataset(
            zarr_path=zarr_path,
            feature_vector_path=feature_path,
            transform=self.transforms,
            target_transform=self.target_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return self.train_dataloader()

    def prepare_data(self):
        # No data downloading or generation needed for static dataset
        pass
