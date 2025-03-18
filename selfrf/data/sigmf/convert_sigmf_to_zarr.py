import os
import json
import numpy as np
import zarr
import tempfile
from sigmf.sigmffile import SigMFFile


def load_sigmf_iq(meta_path: str, data_path: str) -> (np.ndarray, dict):
    """Load SigMF metadata and IQ data with improved error handling."""
    try:
        # Verify file exists and has content
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        # Read the JSON content directly
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_content = json.load(f)
            print(f"📄 Successfully loaded JSON metadata")

        # Map SigMF datatypes to NumPy datatypes
        dtype_mapping = {
            'cf32_le': np.complex64,
            'cf64_le': np.complex128,
            'rf32_le': np.float32,
            'rf64_le': np.float64
        }

        # Get datatype from metadata
        sigmf_dtype = meta_content["global"]["core:datatype"]
        if sigmf_dtype not in dtype_mapping:
            raise ValueError(f"Unsupported datatype: {sigmf_dtype}")

        numpy_dtype = dtype_mapping[sigmf_dtype]
        print(f"📊 Converting {sigmf_dtype} to {numpy_dtype.__name__}")

        # Load and validate IQ data
        iq_data = np.fromfile(data_path, dtype=numpy_dtype)

        if iq_data.size == 0:
            raise ValueError(f"No IQ data found in {data_path}")

        # Extract metadata
        sample_rate = meta_content["global"]["core:sample_rate"]
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")

        processed_metadata = {
            "num_iq_samples_dataset": iq_data.shape[0],
            "impairment_level": 2,
            "fft_size": 512,
            "sample_rate": float(sample_rate),
            "datatype": sigmf_dtype
        }

        print(f"📈 Loaded {iq_data.shape[0]} IQ samples")
        return iq_data, processed_metadata

    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {meta_path}")
        print(f"Error details: {str(e)}")
        with open(meta_path, 'rb') as f:
            print(f"File content (first 100 bytes):")
            print(f.read(100))
        raise
    except Exception as e:
        print(f"❌ Error processing {meta_path}")
        print(f"Error details: {str(e)}")
        raise


def save_iq_to_zarr(iq_data: np.ndarray, metadata: dict, zarr_dir: str):
    """Save IQ data and metadata to Zarr format with optimized chunking."""
    # Ensure directory exists
    os.makedirs(zarr_dir, exist_ok=True)

    # Create new store (overwrite if exists)
    store_path = os.path.join(zarr_dir, "data.zarr")
    if os.path.exists(store_path):
        print(f"🗑️ Removing existing zarr store at {store_path}")
        import shutil
        shutil.rmtree(store_path)

    store = zarr.DirectoryStore(store_path)
    root = zarr.group(store)

    # Calculate optimal chunk size
    total_samples = iq_data.shape[0]
    item_size = iq_data.dtype.itemsize
    target_chunk_size = 16 * 1024 * 1024  # 16MB chunks

    # Calculate chunk shape
    chunk_samples = min(total_samples, target_chunk_size // item_size)
    chunk_shape = (chunk_samples,)
    chunk_size_mb = (chunk_samples * item_size) / (1024 * 1024)

    print(f"💾 Using chunks of {chunk_size_mb:.1f}MB ({chunk_shape} samples)")

    # Create dataset with optimized chunking
    root.create_dataset("iq_samples",
                        data=iq_data,
                        dtype=iq_data.dtype,
                        chunks=chunk_shape,
                        compression='blosc',
                        compression_opts={'cname': 'lz4', 'clevel': 5})

    # Save metadata
    root.attrs["metadata"] = metadata
    print(f"✅ Saved {len(iq_data)} IQ samples ({iq_data.nbytes / 1e6:.1f}MB) "
          f"to {store_path}")


def process_all_sigmf(input_folder: str, output_folder: str):
    """Process all SigMF files in the input folder with improved error handling."""
    if not os.path.exists(input_folder):
        print(f"❌ Input folder {input_folder} does not exist!")
        return

    meta_files = [f for f in os.listdir(
        input_folder) if f.endswith('.sigmf-meta')]
    if not meta_files:
        print(f"❌ No SIGMF datasets found in {input_folder}!")
        return

    print(f"📁 Found {len(meta_files)} SIGMF datasets")

    for meta_file in meta_files:
        dataset_name = meta_file.replace('.sigmf-meta', '')
        meta_path = os.path.join(input_folder, meta_file)
        data_path = os.path.join(input_folder, f"{dataset_name}.sigmf-data")

        print(f"🔄 Processing {dataset_name}...")
        print(f"  Meta file: {meta_path}")
        print(f"  Data file: {data_path}")

        if not os.path.exists(data_path):
            print(f"⚠️ Skipping {dataset_name}, missing .sigmf-data file!")
            continue

        try:
            zarr_dir = os.path.join(output_folder, dataset_name, "train")
            iq_data, processed_metadata = load_sigmf_iq(meta_path, data_path)
            save_iq_to_zarr(iq_data, processed_metadata, zarr_dir)
        except Exception as e:
            print(f"❌ Failed to process {dataset_name}")
            print(f"Error: {str(e)}")
            continue

    print("✅ Conversion complete!")


if __name__ == '__main__':
    # Build paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust the relative path based on your project structure.
    input_folder = os.path.abspath(
        os.path.join(script_dir, "../../../datasets/SIGMF"))
    output_folder = os.path.abspath(os.path.join(
        script_dir, "../../../datasets/iqdm_full_spectrum"))

    process_all_sigmf(input_folder, output_folder)
