
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import Dataset

from torchsig.signals import Signal
from torchsig.transforms.base_transforms import Transform

from selfrf.transforms.extra.torchsig_legacy_transforms import Identity
from selfrf.finetuning.detection.detectron2.mapper import _signal_to_coco_annotation


class RFCOCODataset(Dataset):
    """Base class for RF signal datasets in COCO format.

    Supports:
    - Narrowband: Single annotation per IQ frame (classification)
    - Wideband: Multiple annotations per IQ frame (detection)

    Directory structure:
        root/
            train/
                iq_0.npy  # IQ samples in complex64 format
                iq_1.npy
                ...
            val/
            annotations/
                instances_train.json  # COCO-styled format annotations
                instances_val.json
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None,
    ) -> None:
        """
        Args:
            root: Dataset root directory
            transform: Optional transform to be applied on IQ samples
        """
        self.root = Path(root)
        self.dataset_name = self.root.name
        self.split = split

        self.transform = transform or Identity()
        self.target_transform = target_transform or Identity()

        with open(self._annotations_json_path(), 'r') as f:
            self.coco = json.load(f)

        # map iq frame id to annotations
        self.mapping: Dict[int, List] = {}
        for ann in self.coco['annotations']:
            iq_frame_id = ann['iq_frame_id']

            if iq_frame_id not in self.mapping:
                self.mapping[iq_frame_id] = []

            self.mapping[iq_frame_id].append(ann)

        # Get all IQ frame info
        self.iq_frames = self.coco['iq_frames']

        # labels (not used right now)
        self.categories: Dict = self.coco['categories']
        self.class_list = [category['name'] for category in self.categories]

    def get_image_detection_dicts(self) -> List[Dict]:
        """Convert RF COCO annotations to detectron training format"""
        detection_records = []
        for iq_frame in self.iq_frames:

            annnotations = self.mapping.get(iq_frame["id"], [])

            for ann in annnotations:
                ann["category_id"] = self.class_list.index(ann["class_name"])

            record = {
                "file_name": str(self.root / self.split / iq_frame["file_name"]),
                "image_id": iq_frame["id"],
                "annotations": annnotations
            }

            detection_records.append(record)
        return detection_records

    def get_evaluation_dicts(self) -> List[Dict]:
        """Convert RF COCO annotations to detectron evaluation format"""
        HEIGHT, WIDTH = 512, 512  # Assume fixed size for now
        detection_records = []
        for iq_frame in self.iq_frames:

            annnotations = self.mapping.get(iq_frame["id"], [])

            signal = Signal(
                metadata=annnotations,
            )

            record = {
                "file_name": str(self.root / self.split / iq_frame["file_name"]),
                "image_id": iq_frame["id"],
                "annotations": _signal_to_coco_annotation(signal, HEIGHT, WIDTH),
                "height": HEIGHT,
                "width": WIDTH,
            }

            detection_records.append(record)
        return detection_records

    def _annotations_json_path(self) -> Path:
        """Returns the path to the COCO-style annotations JSON file."""
        return self.root / "annotations" / f"instances_{self.split}.json"

    def _data_dir(self) -> Path:
        """Returns the path to the data directory."""
        return self.root / self.split

    def __len__(self) -> int:
        """Returns the total number of IQ frames in the dataset."""
        return len(self.iq_frames)

    def get_data(self, idx: int) -> np.ndarray:
        """Load IQ samples from .npy file"""
        frame_info = self.iq_frames[idx]
        iq_path = self._data_dir() / frame_info['file_name']
        return np.load(iq_path)

    def get_meta(self, idx: int) -> Union[Dict, List[Dict]]:
        """Get target annotations for IQ frame"""
        return self.mapping[idx]

    def get_data_and_meta(self, idx: int) -> Tuple[np.ndarray, Union[Dict, List[Dict]]]:
        """Get IQ samples and target annotations for IQ frame"""
        return self.get_data(idx), self.get_meta(idx)

    def apply_target_transfrom(self, view: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single view of the signal data"""
        # apply target transform, to get target in desired format
        target = self.target_transform(view["metadata"])
        samples = view["data"]["samples"]
        return samples, target

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, List[np.ndarray]], Dict]:
        """Get item with support for multi-view transforms

        Returns:
            Tuple containing:
                - Single np.ndarray or List[np.ndarray] for multi-view
                - Dict Target 
        """
        iq_data, meta = self.get_data_and_meta(idx)

        # Create Signal object
        signal = Signal(
            data=iq_data,
            metadata=meta,
        )

        # Apply transform on data and meta
        transformed = self.transform(signal)

        if isinstance(transformed, list):
            samples, targets = zip(*[self.apply_target_transfrom(view)
                                   for view in transformed])

            # Assume all views have the same target,
            # as multi-view transforms are only used for self-supervised learning
            return list(samples), targets[0]

        samples, target = self.apply_target_transfrom(transformed)
        return samples, target
