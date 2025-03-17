import os
import json
import numpy as np
import zarr
from numcodecs import JSON


def load_sigmf_iq(meta_path: str, data_path: str) -> np.ndarray:
    """Load IQ samples from SIGMF meta and data files."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    dtype = np.dtype(np.complex64)  # Default to complex64
    iq_data = np.fromfile(data_path, dtype=dtype)

    # Extract necessary metadata
    sample_rate = meta["global"].get(
        "core:sample_rate", 100000)  # Default to 100k if missing

    # Ensure correct metadata format
    metadata = [
        {
            "class_index": 0,  # Placeholder for SSL training
            "start": 0,
            "stop": len(iq_data),
            "lower_freq": 0,
            "upper_freq": sample_rate / 2,  # Assume Nyquist limit
            "sample_rate": sample_rate
        }
    ]

    return iq_data, metadata


def save_iq_to_zarr(iq_data: np.ndarray, metadata: dict, zarr_dir: str):
    """Save IQ data & metadata into Zarr format with chunking."""
    os.makedirs(zarr_dir, exist_ok=True)

    store = zarr.DirectoryStore(os.path.join(zarr_dir, "data.zarr"))
    root = zarr.group(store)

    # ✅ Save IQ Samples
    root.create_dataset("iq_samples", data=iq_data,
                        dtype='complex64', chunks=(1000,))

    # ✅ Convert metadata dictionary to JSON string
    metadata_json = json.dumps(metadata)

    # ✅ Save metadata as a JSON-encoded string in Zarr with an object codec
    root.create_dataset("metadata", data=[metadata_json], dtype=object,
                        object_codec=JSON())

    print(
        f"✅ Saved {len(iq_data)} IQ samples + metadata to {zarr_dir}/data.zarr")


def process_all_sigmf(input_folder, output_folder):
    """Convert all SIGMF datasets in a folder to Zarr format with corrected metadata."""
    meta_files = [f for f in os.listdir(
        input_folder) if f.endswith('.sigmf-meta')]

    if not meta_files:
        print("❌ No SIGMF datasets found!")
        return

    for meta_file in meta_files:
        dataset_name = meta_file.replace('.sigmf-meta', '')
        meta_path = os.path.join(input_folder, meta_file)
        data_path = os.path.join(input_folder, dataset_name + ".sigmf-data")

        if not os.path.exists(data_path):
            print(f"⚠️ Skipping {dataset_name}, missing .sigmf-data file!")
            continue

        zarr_dir = os.path.join(output_folder, dataset_name, "train")
        print(f"🚀 Processing {dataset_name}...")

        iq_data, metadata = load_sigmf_iq(meta_path, data_path)
        save_iq_to_zarr(iq_data, metadata, zarr_dir)

    print("✅ All SIGMF datasets converted to Zarr!")


# Run conversion
input_folder = "c:/Users/SIGENCE/Documents/Lea"  # Where SIGMF files are stored
# Where converted Zarr datasets will be stored
output_folder = "datasets/iqdm_full_spectrum"

process_all_sigmf(input_folder, output_folder)
