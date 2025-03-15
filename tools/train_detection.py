import argparse
import os

from detectron2.utils.logger import setup_logger

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, print_config
from selfrf.finetuning.detection.detectron2.inference import inference_dataset
from selfrf.finetuning.detection.detectron2.register import register_dataset
from selfrf.finetuning.detection.detectron2.lazy_trainer import do_train_lazy
from selfrf.finetuning.detection.detectron2.trainer import do_train
from selfrf.finetuning.detection.detectron2.visualizer import visualize_dataset

setup_logger()


VISUALIZE = False
VISUALIZE_N_SAMPLES = 100
INFERENCE = False


def train(config: Detectron2Config):
    """Register datasets with detectron2."""
    # Check if root is absolute
    if not config.not_absolute_root:  # weired double negation
        config.root = os.path.abspath(config.root)

    register_dataset(config)

    if VISUALIZE:
        visualize_dataset(config, VISUALIZE_N_SAMPLES)

    if INFERENCE:
        inference_dataset(config, 0.3, VISUALIZE_N_SAMPLES)
        return

    if config.model_type.is_lazy_config:
        do_train_lazy(config)
    else:
        do_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train(config)
