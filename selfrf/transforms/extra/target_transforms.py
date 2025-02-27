from typing import Any, Dict, List, Union
import numpy as np
from typing import Any, List
from torchsig.signals.signal_types import SignalMetadata
from torchsig.transforms.target_transforms import TargetTransform


class ConstantTargetTransform(TargetTransform):

    def __init__(self, constant: Any) -> None:
        super().__init__()
        self.constant = constant

    def __call__(self, metadata: Any) -> Any:
        return self.constant


class BBOXLabel(TargetTransform):
    """
    Creates BBOX format annotations from signal metadata in center format (xcycwh)

    Format: [cid, x_center, y_center, width, height] where all values are normalized (0-1)
    """
    output_list = ["list", "dict"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.required_metadata = ["class_index", "start",
                                  "bandwidth", "center_freq", "sample_rate"]

        self.targets_metadata = ["bbox"]

    def __apply__(self, metadata):
        class_index = metadata["class_index"]
        # normalized to width of sample
        width = metadata["duration"]
        # normalize bandwidth with sample rate
        height = metadata["bandwidth"] / metadata["sample_rate"]
        x_center = metadata["start"] + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for image coordinates is upper left,
        # but RF coordinates have (0,0) at lower left
        y_center = 1 - ((metadata["sample_rate"] / 2.0) +
                        metadata["center_freq"]) / metadata["sample_rate"]

        # Create bbox in xcycwh format with class_id as first element
        bbox = [int(class_index), float(x_center), float(
            y_center), float(width), float(height)]

        # Store bbox information
        metadata["bbox"] = bbox

        return metadata
