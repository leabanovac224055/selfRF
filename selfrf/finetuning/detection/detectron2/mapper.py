import copy
import numpy as np
import torch

from detectron2.data import detection_utils


def mapper(dataset_dict):
    # it will be modified by code below
    dataset_dict = copy.deepcopy(dataset_dict)

    # read image and store as torch tensor
    image = detection_utils.read_image(dataset_dict["file_name"], format="L")
    # convert to writable array
    image_shape = image.shape[:2]  # (h, w, c) -> (h, w)

    # Create a writable copy of the image
    image = np.array(image, copy=True, dtype=np.float32)

    # transform the image to tensor (c, h, w)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))

    # normalize the image to [0, 1]
    normalized_image = torch.as_tensor(image) / 255.0
    dataset_dict["image"] = normalized_image

    # annotations to detectron2 instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        dataset_dict["annotations"], image_size=image_shape)

    return dataset_dict
