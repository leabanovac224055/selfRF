import numcodecs
import os
import json
import zarr
import numpy as np
import random
import yaml


def extract_wideband_segments(iq_data, segment_size=65536, step_size=32768, num_signals_min=1, num_signals_max=5):
    """Extract wideband segments containing 1-5 signals."""
    widebands = []
    for start in range(0, len(iq_data) - segment_size, step_size):
        segment = iq_data[start:start + segment_size]

        # Randomly determine how many signals to keep (1 to 5)
        num_signals = random.randint(num_signals_min, num_signals_max)

        # Simulate having fewer signals by zeroing out some random parts
        zero_indices = np.random.choice(
            segment.size, segment.size - (segment.size // num_signals), replace=False)
        segment[zero_indices] = 0

        widebands.append(segment)

    return np.array(widebands)


def extract_narrowband_segments(iq_data, segment_size=8192, step_size=4096):
    """Extract narrowband signals (single signal per segment)."""
    narrowbands = []
    for start in range(0, len(iq_data) - segment_size, step_size):
        segment = iq_data[start:start + segment_size]
        narrowbands.append(segment)
    return np.array(narrowbands)


def split_train_val(data, train_ratio=0.9):
    """Splits the extracted data into train/val based on the given ratio."""
    if len(data) == 0:
        return [], []
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_to_zarr(data, metadata, save_path, dataset_name):
    """Save extracted data and metadata to a Zarr file."""
    if len(data) == 0:
        print(f"⚠️ Skipping {dataset_name}, no extracted samples found!")
        return

    os.makedirs(save_path, exist_ok=True)
    store = zarr.DirectoryStore(os.path.join(save_path, "data.zarr"))
    root = zarr.group(store)

    # ✅ Save IQ samples
    root.create_dataset(dataset_name, data=data,
                        dtype='complex64', chunks=(1000, data.shape[1]))

    # ✅ Save metadata as JSON string
    metadata_json = json.dumps(metadata)  # Convert dict to JSON string
    root.create_dataset(
        "metadata",
        # Store as an object array
        data=np.array([metadata_json], dtype=object),
        object_codec=numcodecs.JSON(),  # Use JSON codec for object storage
    )

    print(f"✅ Saved {len(data)} samples & metadata to {save_path}/data.zarr")


def generate_metadata(save_path, dataset_type):
    """Generate TorchSig-compatible YAML metadata files."""
    dataset_info = {
        "dataset_name": dataset_type,
        "description": f"{dataset_type} dataset extracted from SIGMF",
        "data_format": "zarr",
        "num_samples": 0  # Will be updated
    }

    writer_info = {
        "writer": "extract_wideband_narrowband.py",
        "version": "1.0",
        "notes": f"Automatically generated for {dataset_type} dataset"
    }

    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "create_dataset_info.yaml"), "w") as f:
        yaml.dump(dataset_info, f)

    with open(os.path.join(save_path, "writer_info.yaml"), "w") as f:
        yaml.dump(writer_info, f)


def process_all_datasets(zarr_root, wideband_root, narrowband_root):
    """Extract widebands & narrowbands from all SIGMF datasets and store in Zarr format."""
    dataset_names = os.listdir(zarr_root)
    if not dataset_names:
        print("❌ No SIGMF Zarr datasets found!")
        return

    all_wideband_segments = []
    all_narrowband_segments = []

    for dataset_name in dataset_names:
        zarr_path = os.path.join(zarr_root, dataset_name, "train/data.zarr")
        if not os.path.exists(zarr_path):
            print(f"⚠️ Skipping {dataset_name}, no Zarr file found!")
            continue

        print(f"🚀 Extracting widebands & narrowbands from {dataset_name}...")

        # ✅ Load Zarr Store
        zarr_store = zarr.open(zarr_path, mode="r")
        iq_data = zarr_store["iq_samples"][:]

        # ✅ Ensure metadata exists
        if "metadata" not in zarr_store:
            print(f"⚠️ Metadata missing for {dataset_name}, skipping!")
            continue

        raw_metadata = zarr_store["metadata"][:]
        print(f"🔍 Raw Metadata for {dataset_name}: {raw_metadata}")

        # ✅ Parse metadata correctly
        try:
            # Convert string to JSON
            metadata_json = json.loads(raw_metadata[0])
            metadata = metadata_json[0] if isinstance(
                metadata_json, list) else metadata_json  # Handle list case
        except Exception as e:
            print(f"⚠️ Error parsing metadata for {dataset_name}: {e}")
            continue

        # ✅ Ensure `class_index` is present
        if "class_index" not in metadata:
            metadata["class_index"] = 0  # Default label

        # ✅ Extract sample rate
        sample_rate = metadata.get("sample_rate", metadata.get(
            "global", {}).get("core:sample_rate", None))
        if sample_rate is None:
            print(f"⚠️ Missing 'sample_rate' for {dataset_name}, skipping!")
            continue

        # ✅ Extract Wideband and Narrowband Segments
        wideband_segments = extract_wideband_segments(iq_data)
        narrowband_segments = extract_narrowband_segments(iq_data)

        all_wideband_segments.extend(wideband_segments)
        all_narrowband_segments.extend(narrowband_segments)

    # ✅ Convert to numpy arrays
    all_wideband_segments = np.array(all_wideband_segments)
    all_narrowband_segments = np.array(all_narrowband_segments)

    # ✅ Split into train/val
    wideband_train, wideband_val = split_train_val(all_wideband_segments)
    narrowband_train, narrowband_val = split_train_val(all_narrowband_segments)

    # ✅ Save to Zarr format with metadata
    save_to_zarr(wideband_train, metadata, os.path.join(
        wideband_root, "train"), "iq_samples")
    save_to_zarr(wideband_val, metadata, os.path.join(
        wideband_root, "val"), "iq_samples")

    save_to_zarr(narrowband_train, metadata, os.path.join(
        narrowband_root, "train"), "iq_samples")
    save_to_zarr(narrowband_val, metadata, os.path.join(
        narrowband_root, "val"), "iq_samples")

    # ✅ Generate metadata files
    generate_metadata(os.path.join(wideband_root, "train"), "wideband")
    generate_metadata(os.path.join(wideband_root, "val"), "wideband")

    generate_metadata(os.path.join(narrowband_root, "train"), "narrowband")
    generate_metadata(os.path.join(narrowband_root, "val"), "narrowband")

    print(
        f"✅ Extracted & saved Widebands: {len(all_wideband_segments)} | Narrowbands: {len(all_narrowband_segments)}")


# Run extraction and storage
zarr_root = "datasets/iqdm_full_spectrum"  # Where full IQ datasets are stored
wideband_root = "datasets/iqdm_wideband"
narrowband_root = "datasets/iqdm_narrowband"

process_all_datasets(zarr_root, wideband_root, narrowband_root)
