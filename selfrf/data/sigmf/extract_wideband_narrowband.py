import numcodecs
import os
import json
import zarr
import numpy as np
import random
import yaml


def extract_wideband_segments(iq_data, segment_size=65536, step_size=32768, num_signals_min=1, num_signals_max=5):
    """Extract wideband segments according to TorchSig specifications."""
    widebands = []
    total_segments = (len(iq_data) - segment_size) // step_size + 1

    print(f"📊 Extracting wideband segments:")
    print(f"  - Total IQ samples: {len(iq_data)}")
    print(f"  - Segment size: {segment_size}")
    print(f"  - Step size: {step_size}")
    print(f"  - Expected segments: {total_segments}")

    for start in range(0, len(iq_data) - segment_size, step_size):
        # Extract segment
        segment = iq_data[start:start + segment_size].copy()

        # Create signal mask (TorchSig style)
        num_signals = random.randint(num_signals_min, num_signals_max)
        signal_width = segment_size // num_signals

        # Zero out regions between signals
        for i in range(num_signals):
            start_idx = i * signal_width
            if i < num_signals - 1:
                # Add random gaps between signals
                gap_size = random.randint(signal_width//4, signal_width//2)
                end_idx = start_idx + signal_width - gap_size
                segment[end_idx:start_idx + signal_width] = 0

        widebands.append(segment)

        if len(widebands) % 1000 == 0:
            print(f"  ↳ Processed {len(widebands)}/{total_segments} segments")

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
    """Save extracted data in TorchSig-compatible Zarr format."""
    if len(data) == 0:
        print(f"⚠️ Skipping {dataset_name}, no extracted samples found!")
        return

    os.makedirs(save_path, exist_ok=True)
    store_path = os.path.join(save_path, "data.zarr")

    # Remove existing store if it exists
    if os.path.exists(store_path):
        print(f"🗑️ Removing existing zarr store at {store_path}")
        import shutil
        shutil.rmtree(store_path)

    store = zarr.DirectoryStore(store_path)
    root = zarr.group(store)

    # TorchSig expects specific chunk sizes
    chunk_size = 1000 if "wideband" in dataset_name else 500

    try:
        # Save IQ samples with TorchSig-compatible chunking
        root.create_dataset(
            "iq_samples",
            data=data,
            dtype='complex64',
            chunks=(chunk_size, data.shape[1]),
            compression='blosc',
            compression_opts={'cname': 'lz4', 'clevel': 5}
        )

        # Create TorchSig metadata
        torchsig_metadata = {
            "dataset_info": {
                "name": "wideband" if "wideband" in dataset_name else "narrowband",
                "type": "iq_samples",
                "num_samples": len(data),
                "sample_rate": metadata.get("sample_rate", 1e6),
                "duration": len(data) / metadata.get("sample_rate", 1e6),
                "segment_size": data.shape[1],
                "impairment_level": metadata.get("impairment_level", 2)
            },
            "signal_info": {
                "class_index": metadata.get("class_index", 0),
                "modulation_type": metadata.get("modulation", "unknown"),
                "snr": metadata.get("snr", 20)
            }
        }

        # Save metadata in TorchSig format
        root.attrs["metadata"] = torchsig_metadata

        print(
            f"✅ Saved {len(data)} samples ({data.nbytes / 1e6:.1f}MB) to {store_path}")

    except Exception as e:
        print(f"❌ Error saving dataset {dataset_name}: {str(e)}")
        raise


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
    # Default metadata in case none is found
    default_metadata = {
        "sample_rate": 1e6,
        "impairment_level": 2,
        "class_index": 0,
        "modulation": "unknown",
        "snr": 20
    }

    dataset_names = os.listdir(zarr_root)
    if not dataset_names:
        print("❌ No SIGMF Zarr datasets found!")
        return

    all_wideband_segments = []
    all_narrowband_segments = []
    metadata = default_metadata.copy()  # Initialize with defaults

    for dataset_name in dataset_names:
        zarr_path = os.path.join(zarr_root, dataset_name, "train/data.zarr")
        if not os.path.exists(zarr_path):
            print(f"⚠️ Skipping {dataset_name}, no Zarr file found!")
            continue

        print(f"🚀 Processing {dataset_name}...")

        try:
            # Load Zarr Store
            zarr_store = zarr.open(zarr_path, mode="r")
            iq_data = zarr_store["iq_samples"][:]

            # Try to get metadata from store
            if "metadata" in zarr_store.attrs:
                stored_metadata = zarr_store.attrs["metadata"]
                metadata.update(stored_metadata)
                print(f"📄 Found metadata: {metadata}")
            else:
                print(f"⚠️ Using default metadata for {dataset_name}")

            # Extract segments
            wideband_segments = extract_wideband_segments(iq_data)
            narrowband_segments = extract_narrowband_segments(iq_data)

            all_wideband_segments.extend(wideband_segments)
            all_narrowband_segments.extend(narrowband_segments)

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
            continue

    if not all_wideband_segments and not all_narrowband_segments:
        print("❌ No segments extracted from any dataset!")
        return

    # Convert to numpy arrays
    all_wideband_segments = np.array(all_wideband_segments)
    all_narrowband_segments = np.array(all_narrowband_segments)

    # Split and save datasets
    wideband_train, wideband_val = split_train_val(all_wideband_segments)
    narrowband_train, narrowband_val = split_train_val(all_narrowband_segments)

    # Save all datasets with metadata
    for data, root, name in [
        (wideband_train, wideband_root, "wideband_train"),
        (wideband_val, wideband_root, "wideband_val"),
        (narrowband_train, narrowband_root, "narrowband_train"),
        (narrowband_val, narrowband_root, "narrowband_val")
    ]:
        if len(data) > 0:
            save_path = os.path.join(
                root, "train" if "train" in name else "val")
            save_to_zarr(data, metadata, save_path, name)
            generate_metadata(
                save_path, "wideband" if "wideband" in name else "narrowband")

    print(f"✅ Extraction complete:")
    print(f"  - Widebands: {len(all_wideband_segments)} segments")
    print(f"  - Narrowbands: {len(all_narrowband_segments)} segments")


# Run extraction and storage
zarr_root = "datasets/iqdm_full_spectrum"  # Where full IQ datasets are stored
wideband_root = "datasets/iqdm_wideband"
narrowband_root = "datasets/iqdm_narrowband"

process_all_datasets(zarr_root, wideband_root, narrowband_root)
