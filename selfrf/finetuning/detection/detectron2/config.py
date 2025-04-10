
from enum import Enum
from typing import Literal, NamedTuple, Optional
from dataclasses import fields, dataclass
import argparse
import torch

from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg, CfgNode


from typing import Optional, NamedTuple
from enum import Enum


class ModelConfig(NamedTuple):
    """Model configuration with an optional config path and unique identifier.

    Attributes:
        path: Path to config file. Only required for non-lazy configs.
        is_lazy: Indicates if model uses LazyConfig system.
        identifier: Unique identifier to differentiate configs.
    """
    path: Optional[str] = None  # Optional for lazy configs
    is_lazy: bool = False
    identifier: str = ""


class ModelType(Enum):
    """Supported model architectures with their config paths."""
    FASTER_RCNN_R50_FPN = ModelConfig(
        path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        is_lazy=False,
        identifier="faster_rcnn_r50_fpn"
    )
    VITDET_VIT_B = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_b"
    )
    VITDET_VIT_L = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_l"
    )
    VITDET_VIT_H = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_h"
    )

    @property
    def config_path(self) -> str:
        """Get config path if available.

        Returns:
            str: Path to config file

        Raises:
            ValueError: If trying to access path for lazy config
        """
        if self.is_lazy_config:
            raise ValueError(
                f"Config path not available for lazy config: {self.name}"
            )
        return self.value.path

    @property
    def is_lazy_config(self) -> bool:
        return self.value.is_lazy

    @classmethod
    def from_string(cls, name: str) -> 'ModelType':
        """Get ModelType enum from string.

        Args:
            name: String representation of model type (e.g., 'vitdet-vit-l')

        Returns:
            ModelType: Corresponding enum value

        Raises:
            ValueError: If model type string is not recognized
        """
        try:
            # Convert name to uppercase and replace hyphens with underscores
            enum_name = name.upper().replace('-', '_')
            model_type = cls[enum_name]
            return model_type
        except KeyError:
            # Show available model types in error message
            valid_types = [str(t) for t in cls]
            raise ValueError(
                f"Unknown model type: '{name}'. "
                f"Valid types are: {', '.join(valid_types)}"
            )


# Default values as constants
DEFAULT_ABSOLUTE_ROOT = False
DEFAULT_MODE = "family_recognition"
DEFAULT_NUM_SAMPLES = 5000

DEFAULT_FORCE_RECREATION = False

DEFAULT_MODEL_TYPE = ModelType.FASTER_RCNN_R50_FPN
DEFAULT_PATH = "wideband_impaired"
DEFAULT_WEIGHTS_PATH = ""
DEFAULT_NUM_CLASSES = 10
DEFAULT_MAX_ITER = 90_000
DEFAULT_BASE_LR = 0.0025
DEFAULT_IMS_PER_BATCH = 8
DEFAULT_CHECKPOINT_PERIOD = 1000
DEFAULT_OUTPUT_DIR = "./train/detection"


@dataclass
class Detectron2Config:
    """Configuration for Detectron2 model training."""
    root: str = ""
    absolute_root: bool = DEFAULT_ABSOLUTE_ROOT
    num_samples: int = DEFAULT_NUM_SAMPLES
    mode: Literal["detection", "recognition",
                  "family_recognition"] = DEFAULT_MODE
    force_recreation: bool = DEFAULT_FORCE_RECREATION
    output_dir: str = DEFAULT_OUTPUT_DIR

    model_type: ModelType = DEFAULT_MODEL_TYPE
    dataset_path: str = DEFAULT_PATH
    weights_path: str = DEFAULT_WEIGHTS_PATH
    num_classes: int = DEFAULT_NUM_CLASSES
    max_iter: int = DEFAULT_MAX_ITER
    base_lr: float = DEFAULT_BASE_LR
    ims_per_batch: int = DEFAULT_IMS_PER_BATCH
    checkpoint_period: int = DEFAULT_CHECKPOINT_PERIOD


def add_detectron2_config_args(parser: argparse.ArgumentParser) -> None:
    """Add Detectron2 specific arguments to parser."""
    parser.add_argument(
        '--root',
        type=str,
        default='',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--absolute-root',
        action='store_true',
        default=DEFAULT_ABSOLUTE_ROOT,
        help='If true, root is absolute. '
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help='Number of samples to use for training'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['detection', 'recognition', 'family_recognition'],
        default=DEFAULT_MODE,
        help='Mode of operation: detection or recognition'
    )
    parser.add_argument(
        '--force-recreation',
        action='store_true',
        default=DEFAULT_FORCE_RECREATION,
        help='Force recreation of coco dataset even if it exists'
    )
    parser.add_argument(
        '--model-type',
        type=ModelType.from_string,
        choices=list(ModelType),
        default=DEFAULT_MODEL_TYPE,
        help='Model architecture type (e.g., vitdet-vit-l, vitdet-vit-b)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=DEFAULT_PATH,
        help='path to dataset directory, relative to root'
    )
    parser.add_argument(
        '--weights-path',
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help='Path to pretrained model weights'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help='Number of classes to detect'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=DEFAULT_MAX_ITER,
        help='Maximum number of training iterations'
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=DEFAULT_BASE_LR,
        help='Base learning rate'
    )
    parser.add_argument(
        '--ims-per-batch',
        type=int,
        default=DEFAULT_IMS_PER_BATCH,
        help='Images per batch'
    )
    parser.add_argument(
        '--checkpoint-period',
        type=int,
        default=DEFAULT_CHECKPOINT_PERIOD,
        help='Checkpoint save frequency'
    )


def print_config(config: Detectron2Config) -> None:
    """Print config in a structured format"""
    print("\nDetectron2 Configuration:")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {field.name}: {value}")


def build_detectron2_config(config: Detectron2Config = Detectron2Config()) -> CfgNode:
    """Get Detectron2 config with custom parameters.

    Args:
        config: Custom configuration parameters

    Returns:
        CfgNode: Detectron2 configuration
    """
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("torchsig_wideband_train",)
    cfg.DATASETS.TEST = ("torchsig_wideband_val",)

    # Model parameters
    cfg.MODEL.WEIGHTS = config.weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [0.0]
    cfg.MODEL.PIXEL_STD = [1.0]

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = config.checkpoint_period

    return cfg
