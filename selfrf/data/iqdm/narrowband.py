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
    """Filters a signal within a given frequency range."""

    # Perform FFT
    fft_samples = np.fft.fft(samples)

    # Generate frequency axis
    freq = np.fft.fftfreq(len(samples), 1/sample_rate)

    # Zero out frequencies outside the desired range
    if f_low < f_high:
        fft_samples[(freq < f_low) | (freq > f_high)] = 0
    else:
        fft_samples[(freq > f_low) | (freq < f_high)] = 0

    # Perform inverse FFT
    filtered_samples = np.fft.ifft(fft_samples)

    return filtered_samples.astype(np.complex64)


def filter_time(samples: np.ndarray[np.complex64], sample_start: int, sample_count: int) -> np.ndarray[np.complex64]:
    """ Filters a signal within a given time range. """
    output = np.zeros_like(samples)
    output[sample_start:sample_start +
           sample_count] = samples[sample_start:sample_start + sample_count]

    # 🔍 Debug print before returning
    print(f"📊 Time Filtered Samples (First 5): {output[:5]}")

    return output


def normalize_annotation_to_frame(annotation: dict, sample_start: int, sample_count: int, center_freq: float) -> dict:
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

    # ✅ Create a normalized copy
    copy = deepcopy(annotation)
    copy[SigMFFile.START_INDEX_KEY] = annotation_start - sample_start
    copy[SigMFFile.LENGTH_INDEX_KEY] = annotation_end - annotation_start

    copy[SigMFFile.FLO_KEY] = annotation[SigMFFile.FLO_KEY] - center_freq
    copy[SigMFFile.FHI_KEY] = annotation[SigMFFile.FHI_KEY] - center_freq

    # ✅ Log for verification
    print(f"✅ Normalized Annotation:\n"
          f"  start={copy[SigMFFile.START_INDEX_KEY]} length={copy[SigMFFile.LENGTH_INDEX_KEY]}\n"
          f"  flo={copy[SigMFFile.FLO_KEY]}  fhi={copy[SigMFFile.FHI_KEY]}")

    return copy


def isolate_signal(samples: np.ndarray[np.complex64], sample_rate: float, sample_start: int, sample_count: int, f_low: float, f_high: float) -> np.ndarray[np.complex64]:
    """
    Isolates a signal within a given frequency range from a sample of signals.

    Parameters:
    samples (np.ndarray[np.complex64]): The input signal samples as a complex64 numpy array.
    sample_rate (float): The sample rate of the input signal.
    sample_start (int): The start index of the samples to isolate relative to the input samples.
    sample_count (int): The sample lenght of the signal.
    f_low (float): The lower bound of the frequency range.
    f_high (float): The upper bound of the frequency range.

    Returns:
    np.ndarray[np.complex64]: The isolated signal samples as a complex64 numpy array.
    """

    filtered_samples = filter_frequency(
        samples, sample_rate, f_low, f_high)
    filtered_samples = filter_time(
        filtered_samples, sample_start, sample_count)
    return filtered_samples


def get_capture_for_annotation(sigmf_file: SigMFFile, annotation: dict) -> dict:
    """Find the corresponding capture for an annotation."""
    annotation_start = annotation[SigMFFile.START_INDEX_KEY]
    captures = sigmf_file.get_captures()

    for capture in captures:
        capture_start = capture.get(SigMFFile.START_INDEX_KEY, 0)
        next_capture_start = float('inf')

        # Find next capture start
        for next_capture in captures:
            next_start = next_capture.get(SigMFFile.START_INDEX_KEY, 0)
            if next_start > capture_start and next_start < next_capture_start:
                next_capture_start = next_start

        # Check if annotation falls within this capture
        if capture_start <= annotation_start < next_capture_start:
            return capture

    # If no specific capture found, return first capture or empty dict
    return captures[0] if captures else {}


def preprocess_narrowband_sigmf_files(filepaths, output_dir, frame_size=4096, isolate=True):
    """
    Preprocesses a list of SigMF files and saves them to a single TorchSig-compatible Zarr dataset.

    This function reads each SigMF file, retrieves its annotations, normalizes them to a frame,
    isolates the signal, and saves the extracted signals into a single `data.zarr` file.

    Args:
        filepaths (List[str]): A list of filepaths to the SigMF files to preprocess.
        output_dir (str): Directory where the processed Zarr dataset will be saved.
        frame_size (int): Size of each extracted frame (default is 4096 for TorchSig compatibility).
        isolate (bool): If True, applies time and frequency isolation.

    Returns:
        None
    """

    # ✅ Remove existing dataset if it exists
    zarr_path = os.path.join(output_dir, "data.zarr")
    if os.path.exists(zarr_path):
        print(f"🗑️ Deleting existing dataset at {zarr_path}...")
        shutil.rmtree(zarr_path)

    os.makedirs(output_dir, exist_ok=True)

    all_signals = []
    metadata = []
    feature_vectors = []
    class_map = {}
    class_counter = 0

    for filepath in tqdm(filepaths, desc="Processing SigMF datasets"):
        sigmf_file = SigMFFile()
        # ✅ Load metadata manually
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata_json = json.load(f)

        sigmf_file.set_metadata(metadata_json)
        sigmf_file.set_data_file(
            filepath.replace(".sigmf-meta", ".sigmf-data"))

        file_length = len(sigmf_file)
        sample_rate = sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        annotations = sigmf_file.get_annotations()

        for index, annotation in enumerate(tqdm(annotations, desc=f"Processing annotations for {os.path.basename(filepath)}")):
            global_sample_count = frame_size
            capture = get_capture_for_annotation(sigmf_file, annotation)

            # Skip signal if end of file is reached
            if file_length < annotation[SigMFFile.START_INDEX_KEY] + frame_size:
                warnings.warn(
                    f"⚠️ Skipping signal {index}: End of file reached")
                continue

            # Check if f_low is greater than f_high a, indicating no frequency offset
            if annotation[SigMFFile.FLO_KEY] >= annotation[SigMFFile.FHI_KEY]:
                warnings.warn(
                    f"⚠️ Skipping signal {index}: Invalid frequency range (lower >= upper) in {filepath}")
                continue

            # Skip partial signals
            if "(partly)" in annotation[SigMFFile.LABEL_KEY]:
                warnings.warn(
                    f"⚠️ Skipping signal {index}: Label contains '(partly)' in {filepath}")
                continue

            # Adjust start position and length to fit the frame
            if annotation[SigMFFile.LENGTH_INDEX_KEY] > frame_size:
                annotation[SigMFFile.LENGTH_INDEX_KEY] = frame_size
                global_sample_start = annotation[SigMFFile.START_INDEX_KEY]
            else:
                global_sample_start = annotation[SigMFFile.START_INDEX_KEY] - (
                    frame_size - annotation[SigMFFile.LENGTH_INDEX_KEY]) // 2

            # ✅ Normalize annotation (baseband shift)
            normalized_annotation = normalize_annotation_to_frame(
                annotation, global_sample_start, global_sample_count, capture.get(
                    "core:frequency", 0.0)
            )

            if normalized_annotation is None:
                print(f"⚠️ Skipping annotation {index}: Out of frame")
                continue

            # ✅ Read and process signal
            global_sample_start = max(0, global_sample_start)
            samples = sigmf_file.read_samples(
                global_sample_start, global_sample_count)

            # ✅ Apply isolation if needed
            if isolate:
                samples = isolate_signal(
                    samples, sample_rate, sample_start=normalized_annotation[
                        SigMFFile.START_INDEX_KEY], sample_count=normalized_annotation[SigMFFile.LENGTH_INDEX_KEY], f_low=normalized_annotation[SigMFFile.FLO_KEY], f_high=normalized_annotation[SigMFFile.FHI_KEY])

            # ✅ Check if signal contains only zeros
            if np.allclose(samples, 0, atol=1e-10):
                print(f"⚠️ Signal {index} is completely silent (all zeros).")
                warnings.warn(
                    f"⚠️ Skipping signal {index}: Contains only zero values")
                continue

            # Skip zero signals
            if np.allclose(samples, 0, atol=1e-10):
                warnings.warn(
                    f"⚠️ Skipping signal {index}: Contains only zero values")
                continue

            # Log valid signals
            print(f"✅ Processing signal {index}: Seems valid!")

            # ✅ Pad or trim to `frame_size`
            if len(samples) > frame_size:
                samples = samples[:frame_size]
            elif len(samples) < frame_size:
                samples = np.pad(
                    samples, (0, frame_size - len(samples)), mode="constant")

            # Store the processed signal
            all_signals.append(samples)

            class_name = normalized_annotation.get("core:label", "unknown")

            if class_name not in class_map:
                class_map[class_name] = class_counter
                class_counter += 1

            class_index = class_map[class_name]

            # ✅ Store metadata in TorchSig-compatible format
            metadata.append({
                "bandwidth": normalized_annotation[SigMFFile.FHI_KEY] - normalized_annotation[SigMFFile.FLO_KEY],
                "center_freq": (normalized_annotation[SigMFFile.FLO_KEY] + normalized_annotation[SigMFFile.FHI_KEY]) / 2,
                "class_index": class_index,
                "class_name": class_name,
                "duration": global_sample_count / sample_rate,
                "duration_in_samples": global_sample_count,
                "lower_freq": normalized_annotation[SigMFFile.FLO_KEY],
                "num_samples": global_sample_count,
                "sample_rate": sample_rate,
                "snr_db": normalized_annotation.get("snr_db", 0.0),
                "start": normalized_annotation[SigMFFile.START_INDEX_KEY] / frame_size,
                "start_in_samples": normalized_annotation[SigMFFile.START_INDEX_KEY],
                "stop": (normalized_annotation[SigMFFile.START_INDEX_KEY] + normalized_annotation[SigMFFile.LENGTH_INDEX_KEY]) / frame_size,
                "stop_in_samples": normalized_annotation[SigMFFile.START_INDEX_KEY] + global_sample_count,
                "upper_freq": normalized_annotation[SigMFFile.FHI_KEY]
            })

            # ✅ Save ORIGINAL (not shifted) metadata for Tower B
            original_flo = annotation[SigMFFile.FLO_KEY]
            original_fhi = annotation[SigMFFile.FHI_KEY]
            original_center_freq = (original_flo + original_fhi) / 2
            original_bandwidth = abs(original_fhi - original_flo)

            feature_vectors.append({
                "original_center_freq": original_center_freq,
                "original_bandwidth": original_bandwidth,
                "duration": global_sample_count / sample_rate,
                "class_index": class_index
            })

    # ✅ Convert signals to a NumPy array
    all_signals = np.array(all_signals, dtype=np.complex64)

    # ✅ Save the dataset to TorchSig-compatible Zarr format
    save_to_single_zarr(zarr_path, all_signals, metadata, frame_size)

    with open(os.path.join(output_dir, "feature_vectors.json"), "w") as f:
        json.dump(feature_vectors, f, indent=2)

    print(f"✅ Saved {len(feature_vectors)} feature vectors.")
    print(f"✅ Saved {len(all_signals)} signals to {zarr_path}")


def save_to_single_zarr(zarr_path, all_signals, metadata, frame_size=4096):
    """
    Saves all extracted narrowband signals into a single Zarr array.

    Args:
        output_path (str): Path to save the Zarr dataset.
        all_signals (np.ndarray): Extracted signals (num_signals, frame_size).
        metadata (list): Metadata in TorchSig format.
        frame_size (int): Frame size (default 4096 for TorchSig compatibility).
    """

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    if all_signals.size == 0:
        warnings.warn("⚠️ No valid signals to save. Skipping Zarr storage.")
        return

    shape = all_signals.shape
    if shape[0] == 0:
        warnings.warn("⚠️ No valid signals. Skipping Zarr file creation.")
        return

    chunk_size = (min(100, shape[0]), frame_size)

    # ✅ Create a single contiguous Zarr array
    zarr_store = zarr.open_array(
        store=zarr_path,
        mode="w",
        shape=all_signals.shape,
        dtype=np.complex64,
        chunks=chunk_size,  # ✅ Match TorchSig chunk size `[100, 4096]`
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
        f"✅ Successfully saved {all_signals.shape[0]} signals to {zarr_path}")


# 🚀 Run preprocessing
input_folder = "/home/sigence/selfRF/datasets/SIGMF"
output_folder = "/home/sigence/selfRF/datasets/NARROWBAND_ZARR"

preprocess_narrowband_sigmf_files(
    filepaths=[os.path.join(input_folder, f) for f in os.listdir(
        input_folder) if f.endswith(".sigmf-meta")],
    output_dir=output_folder,
    frame_size=4096,
    isolate=True
)
