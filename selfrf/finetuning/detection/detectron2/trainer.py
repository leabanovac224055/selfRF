import os
import json
from datetime import datetime

import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config

from .mapper import mapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=mapper
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for the given dataset.
        This uses the COCOEvaluator for object detection evaluation.

        Args:
            cfg: Detectron2 config
            dataset_name: Dataset name (e.g., "torchsig_wideband_val")
            output_folder: Output directory for evaluation files
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            tasks=("bbox",),  # Only evaluate bounding boxes
            use_fast_impl=True
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns the test loader for a dataset.
        Uses the same mapper as the train loader to ensure consistency.
        """
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=mapper
        )


def do_train(config: Detectron2Config):
    """
    Train a Detectron2 model with the given configuration.
    """

    cfg = build_detectron2_config(config)

    # Create a uniquely named output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config.output_dir,
        f"run_{timestamp}"  # create a unique directory for each run
    )

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    run_info = {
        "timestamp": timestamp,
        "config": config.__dict__,
        "output_dir": output_dir
    }

    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    trainer = Trainer(cfg)

    model = trainer.model

    # Load SSL weights if specified
    if config.weights_path:
        model = load_ssl_weights(model, config.weights_path)
        # Update trainer model
        trainer.model = model
        print(f"SSL weights loaded from {config.weights_path}")

    # Load checkpoint if available
    trainer.resume_or_load(resume=False)

    # Run training
    print(f"Starting training with output to: {output_dir}")
    trainer.train()

    # Print location of final model
    print(f"Training complete. Model saved to: {output_dir}")


def load_ssl_weights(model, checkpoint_path):
    """
    Load weights from a self-supervised learning checkpoint 
    into a Detectron2 model.

    Args:
        model: Detectron2 model
        checkpoint_path: Path to SSL checkpoint

    Returns:
        model: Updated model with loaded weights
    """
    print(f"Loading SSL weights from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract state_dict if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Create new state dict with correct key mapping
    mapped_state_dict = {}

    # Key mapping: SSL checkpoint → Detectron2 model
    for k, v in checkpoint.items():
        # Skip non-backbone parameters
        if not k.startswith('resnet.') and not k.startswith('backbone.'):
            continue

        # Handle different prefix cases
        if k.startswith('resnet.'):
            # Map SSL keys to Detectron2 keys
            new_key = k.replace('resnet.', 'backbone.bottom_up.')
            mapped_state_dict[new_key] = v
        elif k.startswith('backbone.'):
            # Already has backbone prefix, just check if bottom_up is needed
            if 'bottom_up' not in k:
                new_key = k.replace('backbone.', 'backbone.bottom_up.')
                mapped_state_dict[new_key] = v
            else:
                # Already has correct format
                mapped_state_dict[k] = v

    # Log statistics
    print(f"Original checkpoint keys: {len(checkpoint)}")
    print(f"Mapped state dict keys: {len(mapped_state_dict)}")

    # Load weights with strict=False to allow partial loading
    missing, unexpected = model.load_state_dict(
        mapped_state_dict, strict=False)

    # Log loading results
    if len(missing) > 0:
        print(f"Missing keys: {len(missing)}")
        print(missing)

    if len(unexpected) > 0:
        print(f"Unexpected keys: {len(unexpected)}")
        print(unexpected)

    return model
