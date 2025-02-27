import copy
from typing import Dict, List, Tuple
import numpy as np

from detectron2.data import detection_utils
from detectron2.structures import BoxMode

from torchsig.signals import Signal
from torchsig.signals.signal_lists import TorchSigSignalLists

from selfrf.transforms.extra.target_transforms import BBOXLabel
from selfrf.pretraining.config.base_config import BaseConfig


FFT_SIZE = 512
class_list = TorchSigSignalLists.all_signals
target_transform = BBOXLabel()


def rfcoco_mapper(dataset_dict: Dict):
    """
    Maps raw dataset dictionary to Detectron2-compatible format by processing IQ data and annotations.

    This function performs the following operations:
    1. Loads IQ data from NumPy array
    2. Applies signal and spectrogram transformations
    3. Converts relative bounding box coordinates to absolute pixel coordinates
    4. Creates Detectron2-compatible annotation format
    5. Generates detection instances

    Args:
        dataset_dict (Dict): Input dictionary containing:
            - file_name: Path to NumPy array with IQ data
            - annotations: List of original annotations with relative coordinates

    Returns:
        Dict: Transformed dictionary containing:
            - image: Transformed spectrogram tensor (1, H, W)
            - height: Image height in pixels
            - width: Image width in pixels
            - annotations: List of Detectron2-compatible annotations with absolute coordinates
            - instances: Detectron2 Instances object with detection targets
    """
    from selfrf.pretraining.factories.transform_factory import TransformFactory
    spectrogram_transform = TransformFactory.create_spectrogram_transform(
        BaseConfig(nfft=FFT_SIZE))

    dataset_dict = copy.deepcopy(dataset_dict)

    # Load NumPy array instead of image
    iq_data = np.load(dataset_dict["file_name"])

    annotations = dataset_dict["annotations"]
    signal = Signal(
        data=iq_data,
        metadata=annotations,
    )

    # apply data transform
    transformed_signal: Signal = spectrogram_transform(signal)
    # (1, H, W)
    spectrogram_tensor: np.ndarray = transformed_signal["data"]["samples"]
    height, width = spectrogram_tensor.shape[1:]

    dataset_dict["image"] = spectrogram_tensor
    dataset_dict["height"] = height
    dataset_dict["width"] = width

    # apply target transform
    coco_annotations = _signal_to_coco_annotation(
        transformed_signal, height, width)

    # overwrite annotations
    dataset_dict["annotations"] = coco_annotations

    # create instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        coco_annotations,
        image_size=(height, width)
    )

    return dataset_dict


def rf_coco_evaluation_mapper(dataset_dict: Dict):
    """
    Only create spectrogram and instances for evaluation dataset.
    Do not overwrite annotations.
    """
    from selfrf.pretraining.factories.transform_factory import TransformFactory
    spectrogram_transform = TransformFactory.create_spectrogram_transform(
        BaseConfig(nfft=FFT_SIZE))

    dataset_dict = copy.deepcopy(dataset_dict)

    # Load NumPy array instead of image
    iq_data = np.load(dataset_dict["file_name"])

    signal = Signal(
        data=iq_data,
        metadata=[],  # empty metadata because we don't need it
    )

    # apply data transform
    transformed_signal: Signal = spectrogram_transform(signal)
    # (1, H, W)
    spectrogram_tensor: np.ndarray = transformed_signal["data"]["samples"]
    height, width = spectrogram_tensor.shape[1:]

    dataset_dict["image"] = spectrogram_tensor

    # create instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        dataset_dict["annotations"],
        image_size=(height, width)
    )

    return dataset_dict


def _convert_to_xywh_abs(bbox, width, height) -> Tuple[int, int, int, int]:
    x_rel, y_rel, w_rel, h_rel = bbox

    # Convert to absolute pixel coordinates
    x_pix = int(x_rel * width)  # width
    y_pix = int(y_rel * height)  # height
    w_pix = int(w_rel * width)
    h_pix = int(h_rel * height)

    return (x_pix, y_pix, w_pix, h_pix)


def _signal_to_coco_annotation(signal: Signal, height: int, width: int) -> List[Dict]:
    """Convert signal metadata to COCO format annotations.

    This function transforms signal metadata into COCO-style annotations, converting
    bounding box coordinates to absolute pixel values and formatting them according
    to COCO dataset specifications.

    Args:
        signal (Signal): Input signal containing metadata with annotations.
            Expected to have 'metadata' key with annotation information.
        height (int): Height of the image/frame in pixels.
        width (int): Width of the image/frame in pixels.

    Returns:
        List[Dict]: List of COCO format annotations. Each annotation contains:
            - id: Original annotation ID
            - image_id: Frame ID from the IQ data
            - category_id: Class label
            - bbox: List of [x, y, width, height] in absolute pixel coordinates
            - bbox_mode: BoxMode.XYWH_ABS indicating absolute pixel coordinates
            - area: Area of bounding box in square pixels
            - iscrowd: Always 0, indicating instance segmentation
    """

    # apply target transform
    annotations = signal["metadata"]

    transformed_targets = target_transform(signal["metadata"])

    labels, boxes = transformed_targets["labels"], transformed_targets["boxes"]
    coco_annotations = []
    for index, annotation in enumerate(annotations):

        # Convert to absolute pixel coordinates
        x_pix, y_pix, w_pix, h_pix = _convert_to_xywh_abs(
            boxes[index],
            width,
            height
        )

        coco_annotations.append({
            "id": annotation["id"],
            "image_id": annotation["iq_frame_id"],
            "category_id": labels[index],
            "bbox": [x_pix, y_pix, w_pix, h_pix],
            "bbox_mode": BoxMode.XYWH_ABS,  # XYWH format
            "area": float(w_pix * h_pix),
            "iscrowd": 0,
        })

    return coco_annotations
