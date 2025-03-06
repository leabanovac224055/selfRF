import os

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

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create trainer with COCO evaluation capabilities
    trainer = Trainer(cfg)

    # Load checkpoint if available
    trainer.resume_or_load(resume=False)

    # Run training
    trainer.train()
