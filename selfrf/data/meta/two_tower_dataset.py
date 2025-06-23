import os
import zarr
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchsig.signals.signal_types import DatasetSignal
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.factories import build_collate_fn, metadata_collate_fn, metadata_collate_fn_eval 
from selfrf.pretraining.utils.enums import TrainingStage


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        feature_vector_path: str,
        label_info_path: str,
        transform=None,
        target_transform=None,
        dataset_metadata=None,
        load_class_name=False,
        training_stage=TrainingStage.SSL,
    ):
        print(f"[DEBUG] Initializing TwoTowerDataset with training stage: {training_stage}")

        # Open IQ data (.zarr)
        self.zarr_data = zarr.open_array(zarr_path, mode='r')
        self.attrs = self.zarr_data.attrs.asdict()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_metadata = dataset_metadata
        self.load_class_name = load_class_name
        self.training_stage = training_stage

        # Load metadata feature vectors from .npy (already scaled)
        self.feature_vectors = np.load(feature_vector_path).astype(np.float32)
        
        with open(label_info_path, 'r') as f:
            self.label_info = json.load(f)

        assert len(self.zarr_data) == len(self.feature_vectors), (
            f"Mismatch between IQ samples ({len(self.zarr_data)}) and "
            f"metadata vectors ({len(self.feature_vectors)})"
        )

    def __len__(self):
        return len(self.zarr_data)

    def __getitem__(self, idx):
        vector_dict = self.label_info[idx]
        metadata_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32)
        label = (
            (vector_dict.get("class_index", 0), vector_dict.get("class_name", "unknown"))
            if self.load_class_name else vector_dict.get("class_index", 0)
        )

        if self.training_stage == TrainingStage.METADATA:
            return metadata_vector, label
        
        iq = self.zarr_data[idx]
        y_dict = self.attrs[str(idx)][0]

        start_in_samples = y_dict.get("start_in_samples", 0)
        duration_in_samples = y_dict.get("duration_in_samples", iq.shape[-1])
        end_in_samples = start_in_samples + duration_in_samples

        mask = torch.zeros(iq.shape[-1], dtype=torch.float32)
        mask[start_in_samples:end_in_samples] = 1.0

        signal = DatasetSignal(data=iq, signals=[y_dict], dataset_metadata=self.dataset_metadata)
        signal.time_mask = mask

        metadata_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32)
        vector_dict = self.label_info[idx]

        if self.load_class_name:
            label = (vector_dict.get("class_index", 0), vector_dict.get("class_name", "unknown"))
        else:
            label = vector_dict.get("class_index", 0)

        iq_tensor = torch.tensor(np.stack([iq.real, iq.imag], axis=0), dtype=torch.float32)
        mask_tensor = mask

        if self.transform:
            output = self.transform(signal)
            if isinstance(output, DatasetSignal):
                iq_tensor = torch.tensor(output.data, dtype=torch.float32)
                mask_tensor = output.time_mask
                view1 = (iq_tensor, mask_tensor)
                view2 = (iq_tensor, mask_tensor)
            elif isinstance(output, tuple) and len(output) == 2:
                view1, view2 = output
            else:
                raise ValueError(f"Unexpected transform output type: {type(output)}")
        else:
            iq_tensor = torch.tensor(iq, dtype=torch.float32)
            mask_tensor = mask
            view1 = (iq_tensor, mask_tensor)
            view2 = (iq_tensor, mask_tensor)

        return ((view1, view2), metadata_vector, label)
    

class TwoTowerDataModule(LightningDataModule):
    def __init__(
        self,
        config: BaseConfig,
        root: str,
        batch_size: int,
        num_workers: int,
        transforms,
        target_transforms
    ):
        super().__init__()
        self.config = config
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.collate_fn_train = build_collate_fn(config, is_val=False)
        self.collate_fn_val   = build_collate_fn(config, is_val=True)
        self.zarr_path = os.path.join(root, "NARROWBAND_ZARR", "data.zarr")
        self.feature_vector_path = os.path.join(root, "NARROWBAND_ZARR", "X_scaled.npy")
        self.label_info_path = os.path.join(root, "NARROWBAND_ZARR", "feature_vectors.json")
        
        self.dataset_metadata = NarrowbandMetadata(
            num_iq_samples_dataset=4096,
            fft_size=512,
            impairment_level=config.impairment_level,
        )

    def setup(self, stage=None):
        # Determine if class_name should be loaded (True during evaluation)
        load_class_name = stage == "validate" or stage == "test"
        print(f"[DEBUG] TwoTowerDataModule setup called. Stage: {stage}, Load class name: {load_class_name}")
        
        # Load the full dataset
        full_dataset = TwoTowerDataset(
            zarr_path=self.zarr_path,
            feature_vector_path=self.feature_vector_path,
            label_info_path=self.label_info_path,
            transform=self.transforms,
            target_transform=self.target_transforms,
            dataset_metadata=self.dataset_metadata,
            load_class_name=load_class_name,
            training_stage=self.config.training_stage

        )

        # Check if we already split the dataset
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            # Split only if not already done
            val_size = int(0.1 * len(full_dataset))
            train_size = len(full_dataset) - val_size
            
            print(f"[DEBUG] Splitting dataset: train size: {train_size}, val size: {val_size}")

            # Split the dataset with a fixed seed for reproducibility
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
        print(f"[DEBUG] Using collate function: {self.collate_fn_val if stage == 'validate' else self.collate_fn_train}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # also important for validation
            collate_fn=self.collate_fn_val,
        )

    def prepare_data(self):
        # No data downloading or generation needed for static dataset
        pass