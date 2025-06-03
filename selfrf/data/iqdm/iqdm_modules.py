import os
import torch
import zarr
import numpy as np
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader, random_split
from torchsig.datasets.datamodules import TorchSigDataModule
from pytorch_lightning import LightningDataModule
from torchsig.signals.signal_types import DatasetSignal
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn



class ZarrNarrowbandDataset(Dataset):
    def __init__(self, zarr_path, transform=None, target_transform=None, dataset_metadata=None):
        self.data = zarr.open_array(zarr_path, mode='r')
        self.metadata = self.data.attrs.asdict()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_metadata = dataset_metadata  # Needed by DatasetSignal

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y_dict = self.metadata[str(idx)][0]
        signal = DatasetSignal(data=x, signals=[y_dict],
                               dataset_metadata=self.dataset_metadata)
        
        if self.transform:
            signal = self.transform(signal)
            
        y = y_dict

        if self.target_transform:
            # 1) start with your single dict in a list
            metas = [y_dict]
            results = []  # one entry per transform

            # 2) apply each metadata‐transform in turn
            for tt in self.target_transform:
                metas = tt(metas)                       # can return list or single item
                if not isinstance(metas, list):
                    metas = [metas]

                # pull out only the fields that tt declares
                out = [
                    tuple(md[field] for field in tt.targets_metadata)
                    for md in metas
                ]
                results.append(out)                     # append that per‐signal list

            # 3) flatten: narrowband ⇒ one metadata ⇒ tuple of each transform’s single output
            #    results is [[(idx,)], [(name,)]]
            y = tuple(item[0] for item in results)
            # now y == (class_index, class_name)

        return signal.data, y


class IQDMNarrowbandDataModule(LightningDataModule):
    def __init__(self, config, root, batch_size, num_workers, transforms, target_transforms):
        super().__init__()
        self.config = config
        self.zarr_path = os.path.join(root, "NARROWBAND_ZARR", "data.zarr")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        self.collate_fn_train = build_collate_fn(config, is_val=False)
        self.collate_fn_val = build_collate_fn(config, is_val=True)

        # Dummy metadata for DatasetSignal (must match your dataset format)
        self.dataset_metadata = NarrowbandMetadata(
            num_iq_samples_dataset=4096,
            fft_size=512,
            impairment_level=config.impairment_level,
        )

    def setup(self, stage=None):
        full_dataset = ZarrNarrowbandDataset(
            zarr_path=self.zarr_path,
            transform=self.transforms,
            target_transform=self.target_transforms,
            dataset_metadata=self.dataset_metadata
        )

        # Check if we already split the dataset
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            val_size = int(0.1 * len(full_dataset))
            train_size = len(full_dataset) - val_size

            # Split the dataset with a fixed seed for reproducibility
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_train   # 🟢 Training: both views
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_val  # 🟢 Validation: first view if online_linear_eval
        )

    def prepare_data(self):
        pass  # No downloading required



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
