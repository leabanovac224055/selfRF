

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config
from selfrf.finetuning.detection.detectron2.register import register_dataset
from selfrf.finetuning.detection.detectron2.mapper import mapper


def inference_dataset(config: Detectron2Config, threshold: float = 0.7, n_samples=100):
    register_dataset(config)
    cfg = build_detectron2_config(config)
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = config.weights_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Build model directly instead of using DefaultPredictor
    model = build_model(cfg)
    model.eval()

    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    metadata = MetadataCatalog.get("torchsig_wideband_train")
    dataset_dicts = DatasetCatalog.get("torchsig_wideband_train")

    output_dir = Path(config.root) / \
        Path(config.dataset_path) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make sure we don't try to sample more than what's available
    n_samples = min(n_samples, len(dataset_dicts))
    samples = random.sample(dataset_dicts, k=n_samples)
    print(f"Visualizing {n_samples} with inference samples to {output_dir}...")

    for d in samples:
        processed_dict = mapper(d)
        img: torch.Tensor = processed_dict["image"]

        # Prepare input in the format expected by the model
        height, width = img.shape[1], img.shape[2]  # Get image dimensions

        # Create input dict in the format expected by Detectron2 models
        inputs = [{
            "image": img,
            "height": height,
            "width": width
        }]

        # Run inference
        with torch.no_grad():
            outputs = model(inputs)[0]

        if outputs["instances"].has("pred_boxes"):
            print(outputs["instances"].pred_boxes)
        # Convert (C,H,W) to (H,W,C)
        img = img.permute(1, 2, 0)

        visualizer = Visualizer(
            (img * 255).to(torch.uint8),
            metadata=metadata,
            scale=1.0,
        )
        out = visualizer.draw_instance_predictions(
            outputs["instances"].to("cpu"))

        # Save visualization
        out_img = out.get_image()
        plt.imsave(
            output_dir / processed_dict["file_name"].split("/")[-1], out_img)
        print(processed_dict["file_name"].split("/")[-1])
