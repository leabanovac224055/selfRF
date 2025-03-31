import os
import json
import zarr
import numpy as np
import shutil
from tqdm import tqdm
from typing import List, Dict, Any
from sigmf import sigmffile
from sigmf.sigmffile import SigMFFile
from copy import deepcopy


def get_capture_for_annotation(sigmf_file: SigMFFile, annotation: dict) -> dict:
    annotation_start = annotation[SigMFFile.START_INDEX_KEY]
    captures = sigmf_file.get_captures()

    for capture in captures:
        capture_start = capture.get(SigMFFile.START_INDEX_KEY, 0)
        next_capture_start = float('inf')

        for next_capture in captures:
            next_start = next_capture.get(SigMFFile.START_INDEX_KEY, 0)
            if next_start > capture_start and next_start < next_capture_start:
                next_capture_start = next_start

        if capture_start <= annotation_start < next_capture_start:
            return capture

    return captures[0] if captures else {}


def normalize_annotation_to_frame(annotation: dict, sample_start: int, sample_count: int, center_freq: float) -> dict:
    annotation_abs_end = annotation[SigMFFile.START_INDEX_KEY] + \
        annotation[SigMFFile.LENGTH_INDEX_KEY]
    frame_end = sample_start + sample_count
    annotation_start = max(sample_start, annotation[SigMFFile.START_INDEX_KEY])
    annotation_end = min(frame_end, annotation_abs_end)

    if annotation_start >= frame_end or annotation_end <= sample_start:
        return None

    copy = deepcopy(annotation)
    copy[SigMFFile.START_INDEX_KEY] = annotation_start - sample_start
    copy[SigMFFile.LENGTH_INDEX_KEY] = annotation_end - annotation_start
    copy[SigMFFile.FLO_KEY] = annotation[SigMFFile.FLO_KEY] - center_freq
    copy[SigMFFile.FHI_KEY] = annotation[SigMFFile.FHI_KEY] - center_freq

    return copy


def preprocess_wideband_sigmf_files(
    filepaths: List[str],
    output_dir: str,
    frame_size: int = 262144,
    step_size: int = 131072,
    chunk_size: int = 100,
    include_empty: bool = False
) -> None:
    zarr_path = os.path.join(output_dir, "data.zarr")
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    os.makedirs(output_dir, exist_ok=True)

    zarr_store = None
    metadata_dict = {}
    frame_index = 0

    # Initialize class mapping
    class_map = {}
    class_counter = 0

    for filepath in tqdm(filepaths, desc="Processing SigMF wideband files"):
        sigmf = sigmffile.fromfile(filepath)
        sample_rate = sigmf.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        file_length = len(sigmf)
        annotations = sigmf.get_annotations()

        for frame_start in range(0, file_length - frame_size, step_size):
            frame_end = frame_start + frame_size

            samples = sigmf.read_samples(frame_start, frame_size)

            is_empty = np.allclose(samples, 0, atol=1e-10)
            if is_empty and not include_empty:
                continue

            if zarr_store is None:
                zarr_store = zarr.open_array(
                    store=zarr_path,
                    mode="w",
                    shape=(1000, frame_size),
                    chunks=(chunk_size, frame_size),
                    dtype=np.complex64,
                    compressor=zarr.Blosc(cname="zstd", clevel=4, shuffle=2)
                )

            if frame_index >= zarr_store.shape[0]:
                zarr_store.resize((zarr_store.shape[0] + 1000, frame_size))

            zarr_store[frame_index, :] = samples

            frame_annotations = []
            for ann in annotations:
                ann_start = ann[SigMFFile.START_INDEX_KEY]
                ann_len = ann[SigMFFile.LENGTH_INDEX_KEY]
                ann_end = ann_start + ann_len

                if ann_start < frame_end and ann_end > frame_start:
                    # ✅ Label and class index assignment
                    class_name = ann.get("core:label", "unknown")
                    if class_name not in class_map:
                        class_map[class_name] = class_counter
                        class_counter += 1
                    class_index = class_map[class_name]

                    capture = get_capture_for_annotation(sigmf, ann)
                    center_freq = capture.get("core:frequency", 0.0)

                    normalized = normalize_annotation_to_frame(
                        ann, frame_start, frame_size, center_freq
                    )

                    if normalized is None:
                        continue

                    entry = {
                        "bandwidth": normalized[SigMFFile.FHI_KEY] - normalized[SigMFFile.FLO_KEY],
                        "center_freq": (normalized[SigMFFile.FLO_KEY] + normalized[SigMFFile.FHI_KEY]) / 2,
                        "class_index": class_index,
                        "class_name": class_name,
                        "duration": frame_size / sample_rate,
                        "duration_in_samples": frame_size,
                        "lower_freq": normalized[SigMFFile.FLO_KEY],
                        "upper_freq": normalized[SigMFFile.FHI_KEY],
                        "num_samples": frame_size,
                        "sample_rate": sample_rate,
                        "snr_db": normalized.get("snr_db", 0.0),
                        "start": normalized[SigMFFile.START_INDEX_KEY] / frame_size,
                        "start_in_samples": frame_start,
                        "stop": normalized[SigMFFile.START_INDEX_KEY] / frame_size + normalized[SigMFFile.LENGTH_INDEX_KEY] / frame_size,
                        "stop_in_samples": frame_end
                    }

                    frame_annotations.append(entry)

            metadata_dict[str(frame_index)] = frame_annotations
            frame_index += 1

    if zarr_store is None:
        print("⚠️ No valid signals found.")
        return

    zarr_store.resize((frame_index, frame_size))
    zarr_store.attrs.update(metadata_dict)
    print(f"✅ Saved {frame_index} wideband frames to {zarr_path}")


# 🚀 Run preprocessing
input_folder = "/home/sigence/selfRF/datasets/SIGMF"
output_folder = "/home/sigence/selfRF/datasets/WIDEBAND_ZARR"

preprocess_wideband_sigmf_files(
    filepaths=[os.path.join(input_folder, f) for f in os.listdir(
        input_folder) if f.endswith('.sigmf-meta')],
    output_dir=output_folder,
    frame_size=262144,
    step_size=131072,
    chunk_size=100
)
