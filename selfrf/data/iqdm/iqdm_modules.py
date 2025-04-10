import os
import torch
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsig.datasets.datamodules import TorchSigDataModule
from pytorch_lightning import LightningDataModule
from selfrf.pretraining.utils import Signal


class TorchSigMetadata:
    """Wraps a metadata dictionary into an attribute-accessible object."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.applied_transforms = []  # ✅ Required for TorchSig transform tracking


class ZarrNarrowbandDataset(Dataset):
    def __init__(self, zarr_path, transform=None, target_transform=None):
        self.data = zarr.open_array(zarr_path, mode='r')
        self.metadata = self.data.attrs.asdict()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # IQ samples (complex64)
        y_dict = self.metadata[str(idx)][0]
        y = y_dict.get("class_index", 0)

        signal = Signal(data=x, metadata=[TorchSigMetadata(y_dict)])

        if self.transform:
            transformed = self.transform(signal)

            # ✅ Extract views from signal.data (MultiViewTransform returns Signal with .data = [x1, x2])
            if not isinstance(transformed.data, list) or len(transformed.data) != 2:
                raise TypeError(
                    f"Expected signal.data to be a list of two views, got: {type(transformed.data)} with len {len(transformed.data)}"
                )

            x1, x2 = transformed.data
        else:
            x1 = x2 = signal.data

        if self.target_transform:
            y = self.target_transform(y)

        return (x1, x2), y


class IQDMNarrowbandDataModule(LightningDataModule):
    def __init__(self, config, root, batch_size, num_workers, transforms, target_transforms, collate_fn):
        super().__init__()
        self.config = config  # ✅ Store the config
        self.zarr_path = os.path.join(root, "NARROWBAND_ZARR", "data.zarr")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.dataset = ZarrNarrowbandDataset(
            self.zarr_path,
            transform=self.transforms,
            target_transform=self.target_transforms
        )
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset  # or split later

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def prepare_data(self):
        pass  # No downloading needed for static Zarr data


class ZarrWidebandDataset(Dataset):
    def __init__(self, zarr_path, transform=None, target_transform=None):
        self.data = zarr.open_array(zarr_path, mode='r')
        self.metadata = self.data.attrs.asdict()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # (C, T) shape or similar
        y_dicts = self.metadata[str(idx)]  # list of annotations
        y_dict = y_dicts[0]  # pick the first for now, or loop/filter if needed
        y = y_dict.get("class_index", 0)

        signal = Signal(data=x, metadata=[TorchSigMetadata(y_dict)])

        if self.transform:
            transformed = self.transform(signal)
            if not isinstance(transformed.data, list) or len(transformed.data) != 2:
                raise TypeError(
                    f"Expected signal.data to be a list of two views, got: {type(transformed.data)} with len {len(transformed.data)}"
                )
            x1, x2 = transformed.data
        else:
            x1 = x2 = signal.data

        if self.target_transform:
            y = self.target_transform(y)

        return (x1, x2), y


class IQDMWidebandDataModule(LightningDataModule):
    def __init__(self, config, root, batch_size, num_workers, transforms, target_transforms, collate_fn):
        super().__init__()
        self.config = config
        self.zarr_path = os.path.join(root, "WIDEBAND_ZARR", "data.zarr")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        print("🧠 Config class:", type(self.config))
        print("🧠 ssl_model:", getattr(self.config, "ssl_model", None))
        self.dataset = ZarrWidebandDataset(
            self.zarr_path,
            transform=self.transforms,
            target_transform=self.target_transforms
        )
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def prepare_data(self):
        pass
