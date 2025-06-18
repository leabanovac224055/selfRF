import os
import json
import zarr
import numpy as np


def extract_feature_vectors_from_zarr(zarr_path: str, output_json_path: str):
    """
    Extracts metadata feature vectors from a TorchSig-compatible Zarr dataset (.zattrs).

    Args:
        zarr_path (str): Path to the Zarr dataset (e.g., `data.zarr`).
        output_json_path (str): Path to save the extracted feature vectors in JSON format.
    """

    # ✅ Open the Zarr array
    zarr_store = zarr.open_array(zarr_path, mode='r')

    # ✅ Load metadata from .zattrs
    raw_metadata = zarr_store.attrs.asdict()

    feature_vectors = []

    for index_str, metadata_list in raw_metadata.items():
        if not isinstance(metadata_list, list) or not metadata_list:
            print(f"⚠️ No metadata for index {index_str}, skipping...")
            continue

        metadata = metadata_list[0]  # TorchSig stores a list with one dict per index

        try:
            original_center_freq = metadata["center_freq"]
            original_bandwidth = metadata["bandwidth"]
            duration_in_samples = metadata["duration_in_samples"]
            sample_rate = metadata["sample_rate"]
            class_index = metadata["class_index"]
            class_name = metadata["class_name"]
        except KeyError as e:
            print(f"⚠️ Missing key {e} in metadata index {index_str}. Skipping.")
            continue

        # ✅ Custom-normalize duration: 1 ms → 1.0
        normalized_duration = (duration_in_samples / sample_rate) / 1e-3

        # ✅ Create normalized feature vector
        feature_vectors.append({
            "center_freq": original_center_freq / 2.5e9,
            "bandwidth": original_bandwidth / 10e6,
            "duration": normalized_duration / 100,
            "class_index": class_index,
            "class_name": class_name
        })

    # ✅ Save feature vectors
    with open(output_json_path, "w") as f:
        json.dump(feature_vectors, f, indent=2)

    print(f"✅ Extracted {len(feature_vectors)} feature vectors to {output_json_path}")


# 🚀 Example usage
zarr_dataset_path = "/home/airbus/selfRF/datasets/NARROWBAND_ZARR-torchsig/data.zarr"
output_json = "/home/airbus/selfRF/datasets/NARROWBAND_ZARR-torchsig/feature_vectors.json"

extract_feature_vectors_from_zarr(zarr_dataset_path, output_json)