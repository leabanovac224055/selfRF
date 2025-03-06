import os
import json
from pathlib import Path
from typing import Literal
import concurrent.futures
from functools import partial

import cv2
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.wideband import StaticWideband
from tqdm import tqdm


def process_sample(idx, dataset, path_to_image_dir):
    """Process a single sample for parallelization"""
    try:
        spectrogram, labels = dataset[idx]
        filename = f"{idx:010d}.png"
        image_path = str(path_to_image_dir / filename)

        # Save the spectrogram as an image
        cv2.imwrite(image_path, spectrogram, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        height, width = spectrogram.shape

        # Prepare image info
        image_info = {
            "id": idx,
            "file_name": filename,
            "width": width,
            "height": height
        }

        # Prepare annotations
        sample_annotations = []
        sample_categories = {}

        for label in labels:
            bbox = label[0]
            class_name = label[1]
            class_id = label[2] + 1  # COCO categories are 1-indexed
            family_name = label[3]

            sample_categories[class_id] = {
                "id": class_id,
                "name": class_name,
                "supercategory": family_name
            }

            # convert bbox to pixel coordinates
            x = bbox[0] * width
            y = bbox[1] * height
            w = bbox[2] * width
            h = bbox[3] * height
            bbox = [x, y, w, h]
            area = w * h

            sample_annotations.append({
                "bbox": bbox,
                "category_id": class_id,
                "area": area
            })

        return image_info, sample_annotations, sample_categories
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None, None, None


def store_spectrograms(
    dataset: StaticWideband,
    path_to_image_dir: Path,
    max_workers: int = 8  # Adjust based on your CPU cores
) -> tuple:
    """Store spectrograms as images and create COCO annotations"""
    images = []
    annotations = []
    categories = {}
    global_annotation_idx = 0  # global index for annotations

    # Create a partial function with fixed parameters
    process_func = partial(process_sample, dataset=dataset,
                           path_to_image_dir=path_to_image_dir)

    # Process samples in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, idx)
                                   : idx for idx in range(len(dataset))}

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing samples"):
            idx = futures[future]
            image_info, sample_annotations, sample_categories = future.result()

            if image_info is None:
                continue

            images.append(image_info)

            # Add annotations with proper IDs
            for ann in sample_annotations:
                ann["id"] = global_annotation_idx
                ann["image_id"] = idx
                ann["iscrowd"] = 0
                annotations.append(ann)
                global_annotation_idx += 1

            # Update categories
            categories.update(sample_categories)

    return images, annotations, categories


def convert_dataset_to_coco(
    dataset: StaticWideband,
    path_to_coco: Path,
    split: Literal["train", "val"],
):
    """Convert dataset to COCO format"""
    print(f"Converting {split} dataset to COCO format at", path_to_coco)

    images_dir = path_to_coco / "images" / split
    annotations_dir = path_to_coco / "annotations"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Initialize COCO JSON structure
    coco_json = {
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Store images
    images, annotations, categories = store_spectrograms(dataset, images_dir)
    coco_json["images"] = images
    coco_json["annotations"] = annotations
    coco_json["categories"] = list(categories.values())

    # Save COCO JSON
    with open(annotations_dir / f"instances_{split}.json", "w") as f:
        json.dump(coco_json, f, indent=4)


def convert_datamodule_to_coco(
    datamodule: WidebandDataModule,
    dataset_path: str,
    force: bool = False,
) -> Path:
    """Convert datamodule to COCO format"""
    path_to_coco = datamodule.root / dataset_path / "coco"

    if not force and path_to_coco.exists():
        print("COCO format already exists at", path_to_coco)
        return path_to_coco

    os.makedirs(path_to_coco, exist_ok=True)
    print("Converting datamodule to COCO format at", path_to_coco)

    convert_dataset_to_coco(datamodule.train, path_to_coco, split="train")
    convert_dataset_to_coco(datamodule.val, path_to_coco, split="val")

    return path_to_coco
