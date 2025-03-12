

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from selfrf.finetuning.detection.detectron2.config import Detectron2Config
from selfrf.finetuning.detection.detectron2.register import register_dataset
from selfrf.finetuning.detection.detectron2.mapper import mapper


def visualize_dataset(config: Detectron2Config, n_samples=100):
    # register_dataset(config)

    metadata = MetadataCatalog.get("torchsig_wideband_train")
    dataset_dicts = DatasetCatalog.get("torchsig_wideband_train")

    output_dir = Path(config.root) / \
        Path(config.dataset_path) / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make sure we don't try to sample more than what's available
    n_samples = min(n_samples, len(dataset_dicts))
    samples = random.sample(dataset_dicts, k=n_samples)
    print(f"Visualizing {n_samples} samples to {output_dir}...")

    for d in samples:
        processed_dict = mapper(d)
        img: torch.Tensor = processed_dict["image"]
        # print min max
        # print(img.min(), img.max())
        # print(img.shape)

        # Convert (C,H,W) to (H,W,C)
        img = img.permute(1, 2, 0)

        img_uint8 = (img).to(torch.uint8)

        visualizer = Visualizer(
            img_uint8,
            metadata=metadata,
            scale=1.0,
        )
        vis = visualizer.draw_dataset_dict(processed_dict)

        # Save visualization
        vis_img = vis.get_image()
        plt.imsave(
            output_dir / processed_dict["file_name"].split("/")[-1], vis_img)
        print(processed_dict["file_name"].split("/")[-1])


if __name__ == "__main__":
    visualize_dataset()
    plt.show()
