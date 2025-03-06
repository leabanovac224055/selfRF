from typing import Any
from torchsig.transforms.target_transforms import TargetTransform


class ConstantTargetTransform(TargetTransform):

    def __init__(self, constant: Any) -> None:
        super().__init__()
        self.constant = constant

    def __call__(self, metadata: Any) -> Any:
        return self.constant


class BBOXLabel(TargetTransform):
    """
    Creates bounding box annotations in XYWH format (top-left corner)

    Format: [x, y, width, height] where:
    - x: left edge of bounding box (normalized 0-1)
    - y: top edge of bounding box (normalized 0-1)
    - width: width of bounding box (normalized 0-1)
    - height: height of bounding box (normalized 0-1)
    """
    output_list = ["list", "dict"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.required_metadata = ["class_index", "start", "stop",
                                  "lower_freq", "upper_freq", "sample_rate"]
        self.targets_metadata = ["bbox"]

    def __apply__(self, metadata):
        # Time domain calculations
        x = metadata["start"]  # Left edge is start time

        # Calculate width (duration in time)
        width = metadata["duration"]

        # Frequency domain calculations
        # Convert frequencies to normalized values [0-1]
        lower_freq_norm = metadata["lower_freq"] / metadata["sample_rate"]
        upper_freq_norm = metadata["upper_freq"] / metadata["sample_rate"]

        # In standard spectrograms, frequency increases UP the y-axis
        # But in image coordinates, y increases DOWN from top (0) to bottom (1)
        # So we need to flip the y-coordinates

        # Top edge of bbox is the upper frequency bound, flipped
        y = 1.0 - upper_freq_norm

        # Height is the difference between upper and lower, in image coordinates
        height = upper_freq_norm - lower_freq_norm

        # Create and store the bounding box
        metadata["bbox"] = [x, y, width, height]

        return metadata
