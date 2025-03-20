import os
import json
import numpy as np
import zarr
import shutil
import warnings
from tqdm import tqdm
from sigmf.sigmffile import SigMFFile
from copy import deepcopy


def filter_frequency(samples: np.ndarray[np.complex64], sample_rate: float, f_low: float, f_high: float) -> np.ndarray[np.complex64]:
    """ Filters a signal within a given frequency range. """
    fft_samples = np.fft.fft(samples)
    freq = np.fft.fftfreq(len(samples), 1/sample_rate)

    # Zero out frequencies outside the desired range
    if f_low < f_high:
        fft_samples[(freq < f_low) | (freq > f_high)] = 0
    else:
        fft_samples[(freq > f_low) | (freq < f_high)] = 0

    filtered_samples = np.fft.ifft(fft_samples)
    return filtered_samples.astype(np.complex64)


def filter_time(samples: np.ndarray[np.complex64], sample_start: int, sample_count: int) -> np.ndarray[np.complex64]:
    """ Filters a signal within a given time range. """
    output = np.zeros_like(samples)
    output[sample_start:sample_start +
           sample_count] = samples[sample_start:sample_start + sample_count]
    return output


def normalize_annotation_to_frame(annotation: dict, sample_start: int, sample_count: int, sample_rate: float) -> dict:
    """ Normalize an annotation to fit a frame and shift to baseband. """
    annotation_abs_end = annotation[SigMFFile.START_INDEX_KEY] + \
        annotation[SigMFFile.LENGTH_INDEX_KEY]

    frame_end = sample_start + sample_count
    annotation_start = max(sample_start, annotation[SigMFFile.START_INDEX_KEY])
    annotation_end = min(frame_end, annotation_abs_end)

    if annotation_start >= frame_end or annotation_end <= sample_start:
        warnings.warn(
            f"⚠️ Skipping annotation {annotation[SigMFFile.LABEL_KEY]}, outside frame")
        return None

    copy = deepcopy(annotation)
    copy[SigMFFile.START_INDEX_KEY] = annotation_start - sample_start
    copy[SigMFFile.LENGTH_INDEX_KEY] = annotation_end - annotation_start

    copy[SigMFFile.FLO_KEY] = annotation[SigMFFile.FLO_KEY] - sample_rate / 2
    copy[SigMFFile.FHI_KEY] = annotation[SigMFFile.FHI_KEY] - sample_rate / 2

    return copy


def save_to_single_zarr(output_path, all_signals, metadata, frame_size=4096):
    """
    Saves all extracted narrowband signals into a single Zarr array.

    Args:
        output_path (str): Path to save the Zarr dataset.
        all_signals (np.ndarray): Array of extracted signals (num_signals, frame_size).
        metadata (list): Metadata in TorchSig format.
        frame_size (int): Size of each extracted frame.
    """
    # ✅ Remove old dataset if it exists
    if os.path.exists(output_path):
        print(f"🗑️ Deleting existing dataset at {output_path}...")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists

    # ✅ Create a single contiguous Zarr array (TorchSig-compatible)
    zarr_store = zarr.open_array(
        os.path.join(output_path, "data.zarr"),  # Single Zarr array
        mode="w",
        shape=all_signals.shape,
        dtype=np.complex64,
        chunks=(100, frame_size),  # ✅ Match TorchSig chunk size `[100, 4096]`
        compressor=zarr.Blosc(cname="zstd", clevel=4, shuffle=2)
    )

    # ✅ Store the entire dataset inside the array
    zarr_store[:] = all_signals

    # ✅ Convert metadata keys to simple indices and store as a list per index
    formatted_metadata = {str(index): [entry]
                          for index, entry in enumerate(metadata)}

    # ✅ Store metadata in `.zattrs` (TorchSig-compatible format)
    zarr_store.attrs.update(formatted_metadata)

    print(
        f"✅ Successfully saved {all_signals.shape[0]} signals to {output_path}/data.zarr")


def process_sigmf_to_zarr(input_folder, output_folder, frame_size=4096):
    """
    Process all SigMF files, extract narrowbands, normalize, and save to a single Zarr file.

    Args:
        input_folder (str): Directory containing SigMF datasets.
        output_folder (str): Where to store converted Zarr files.
        frame_size (int): Size of each extracted frame (TorchSig uses 4096).
    """
    meta_files = [f for f in os.listdir(
        input_folder) if f.endswith(".sigmf-meta")]
    if not meta_files:
        print("❌ No SigMF metadata files found!")
        return

    all_signals = []
    combined_metadata = []  # Store metadata as a **list** for proper formatting

    for meta_file in tqdm(meta_files, desc="Processing SIGMF datasets"):
        dataset_name = os.path.splitext(meta_file)[0]
        meta_path = os.path.join(input_folder, meta_file)
        data_path = os.path.join(input_folder, f"{dataset_name}.sigmf-data")

        print(f"🔄 Processing {dataset_name}...")

        try:
            # Load SigMF metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_content = json.load(f)

            signal = SigMFFile()
            signal.set_metadata(meta_content)
            signal.set_data_file(data_path)

            sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
            annotations = signal.get_annotations()

            for index, annotation in enumerate(annotations):
                lower_freq = annotation["core:freq_lower_edge"]
                upper_freq = annotation["core:freq_upper_edge"]

                # 🚨 Skip invalid entries where bandwidth = 0
                if lower_freq == upper_freq:
                    print(
                        f"⚠️ Skipping annotation {index}: Fixed frequency signal at {lower_freq} Hz (bandwidth = 0)")
                    continue  # Skip this annotation

                start = annotation["core:sample_start"]
                count = annotation["core:sample_count"]
                signal_data = signal.read_samples(start, count)

                # ✅ Apply **time-domain isolation**
                signal_data = filter_time(signal_data, 0, count)

                # ✅ Apply **frequency-domain isolation**
                signal_data = filter_frequency(
                    signal_data, sample_rate, lower_freq, upper_freq)

                # ✅ Apply **baseband normalization**
                annotation = normalize_annotation_to_frame(
                    annotation, start, count, sample_rate)

                # Skip if normalization failed
                if annotation is None:
                    continue

                # Pad or trim to FRAME_SIZE
                if len(signal_data) > frame_size:
                    signal_data = signal_data[:frame_size]
                elif len(signal_data) < frame_size:
                    signal_data = np.pad(
                        signal_data, (0, frame_size - len(signal_data)), mode="constant")

                all_signals.append(signal_data)

                # ✅ Store metadata in TorchSig format
                combined_metadata.append({
                    "bandwidth": upper_freq - lower_freq,
                    "center_freq": (upper_freq + lower_freq) / 2,
                    "class_index": annotation.get("class_index", -1),
                    "class_name": annotation.get("core:label", "unknown"),
                    "duration": count / sample_rate,
                    "duration_in_samples": count,
                    "lower_freq": lower_freq,
                    "num_samples": count,
                    "sample_rate": sample_rate,
                    "snr_db": annotation.get("snr_db", 0.0),
                    "start": start / len(signal),
                    "start_in_samples": start,
                    "stop": (start + count) / len(signal),
                    "stop_in_samples": start + count,
                    "upper_freq": upper_freq
                })

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")

    # Convert list to NumPy array
    all_signals = np.array(all_signals, dtype=np.complex64)

    # ✅ Save everything to a single Zarr file
    save_to_single_zarr(output_folder, all_signals,
                        combined_metadata, frame_size=4096)


# 🚀 Run preprocessing with correct TorchSig frame size
process_sigmf_to_zarr(
    input_folder="/home/sigence/selfRF/datasets/SIGMF",
    output_folder="/home/sigence/selfRF/datasets/NARROWBAND_ZARR"
)
