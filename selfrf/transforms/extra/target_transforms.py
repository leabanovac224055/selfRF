from typing import Any
from torchsig.transforms.target_transforms import TargetTransform


class Identity(TargetTransform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, metadata):
        return metadata


class BBOXLabel(TargetTransform):
    """
    Adds an XYHW_label to a signal, in the form of a tuple (class_index, x_min, y_min, width, height),
    where coordinates are normalized between 0 and 1.

    Attributes:
        output (str, optional): Structure to aggregate labels ("dict", "list"). Defaults to "list".
    """

    output_list = ["list", "dict"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Include "duration" since it's used in __apply__
        self.required_metadata = [
            "class_index", "start", "duration", "bandwidth", "center_freq", "sample_rate"]
        self.targets_metadata = ["bbox"]

    def __apply__(self, metadata):
        # Extract required metadata
        # X_min is the starting time
        x_min = metadata["start"]
        # Width is the duration, normalized
        width = metadata["duration"]
        # Height is bandwidth normalized by sample rate
        height = metadata["bandwidth"] / metadata["sample_rate"]
        # Compute y_center as in YOLO for consistency
        y_center = 1 - ((metadata["sample_rate"] / 2.0) +
                        metadata["center_freq"]) / metadata["sample_rate"]
        # Y_min is the top edge: y_center - height / 2
        y_min = y_center - height / 2
        # Create the XYHW label tuple
        xyhw_label = (x_min, y_min, width, height)
        # Add to metadata
        metadata["bbox"] = xyhw_label

        return metadata


class ConstantSignalName(TargetTransform):
    """
    Adds a constant signal name to the metadata.
    """

    def __init__(self, signal_name: str, **kwargs):
        super().__init__(**kwargs)
        self.signal_name = signal_name
        self.targets_metadata = ["const_signal_name"]

    def __apply__(self, metadata):
        metadata["const_signal_name"] = self.signal_name
        return metadata


class ConstantSignalIndex(TargetTransform):
    """
    Adds a constant signal index to the metadata.
    """

    def __init__(self, index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.targets_metadata = ["const_signal_index"]
        self.index = index

    def __apply__(self, metadata):
        metadata["const_signal_index"] = self.index
        return metadata


class ConstantFamilyName(TargetTransform):
    """
    Adds a constant family name to the metadata.
    """

    def __init__(self, family_name: str, **kwargs):
        super().__init__(**kwargs)
        self.family_name = family_name
        self.targets_metadata = ["const_family_name"]

    def __apply__(self, metadata):
        metadata["const_family_name"] = self.family_name
        return metadata
