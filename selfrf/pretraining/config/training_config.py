import argparse
from dataclasses import dataclass

from selfrf.pretraining.utils.enums import SSLModelType, MetadataModelType, TrainingStage
from .base_config import BaseConfig, add_base_config_args, parse_base_config

DEFUALT_ONLINE_LINEAR_EVAL = False
DEFAULT_SSL_MODEL = SSLModelType.BYOL
DEFAULT_TRAINING_PATH = './train'
DEFAULT_NUM_EPOCHS = 100
DEFAULT_TWO_TOWER_NUM_EPOCHS = 50
SSL_MODEL_MAP = {model_type.value: model_type for model_type in SSLModelType}
DEFAULT_TRAIN_TWO_TOWER = False

DEFAULT_TRAINING_STAGE = TrainingStage.SSL

# MoCo-v3 specific parameters
DEFAULT_PROJECTION_DIM = 256
DEFAULT_MLP_DIM = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MOMENTUM = 0.999
DEFAULT_QUEUE_SIZE = 65536

DEFAULT_METADATA_MODEL = MetadataModelType.MLP

@dataclass
class TrainingConfig(BaseConfig):
    online_linear_eval: bool = DEFUALT_ONLINE_LINEAR_EVAL
    ssl_model: SSLModelType = DEFAULT_SSL_MODEL
    training_path: str = DEFAULT_TRAINING_PATH
    num_epochs: int = DEFAULT_NUM_EPOCHS
    two_tower_num_epochs: int = DEFAULT_TWO_TOWER_NUM_EPOCHS

    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    
    # MoCo-v3 specific parameters 
    projection_dim: int = DEFAULT_PROJECTION_DIM
    mlp_dim: int = DEFAULT_MLP_DIM
    temperature: float = DEFAULT_TEMPERATURE
    momentum: float = DEFAULT_MOMENTUM
    queue_size: int = DEFAULT_QUEUE_SIZE
    
    metadata_model: MetadataModelType = DEFAULT_METADATA_MODEL


def add_training_config_args(parser: argparse.ArgumentParser) -> None:
    add_base_config_args(parser)
    parser.add_argument(
        '--online-linear-eval',
        type=lambda x: x.lower() == 'true',
        default=DEFUALT_ONLINE_LINEAR_EVAL
    )
    parser.add_argument(
        "--ssl-model",
        type=lambda x: SSLModelType.from_string(x),
        default=SSLModelType.BYOL,
        choices=list(SSLModelType),
        help=f"SSL model to use for pretraining {[model_type.value for model_type in SSLModelType]}"
    )
    parser.add_argument(
        '--training-path',
        type=str,
        default=DEFAULT_TRAINING_PATH
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=DEFAULT_NUM_EPOCHS
    )
    parser.add_argument(
        '--two-tower-num-epochs',
        type=int,
        default=DEFAULT_TWO_TOWER_NUM_EPOCHS,
        help="Number of epochs for two-tower phase"
    )
    parser.add_argument(
        '--training-stage',
        type=lambda x: TrainingStage(x.lower()),
        choices=list(TrainingStage),
        default=DEFAULT_TRAINING_STAGE,
        help="Specify the training stage: ssl, metadata, or fusion"
    )


def parse_training_config() -> TrainingConfig:
    """Parse command line arguments into a TrainingConfig object.

    Creates a base config first, then builds training config from it.
    """
    # Create parser with description
    parser = argparse.ArgumentParser(description="Training Config")
    add_training_config_args(parser)

    # First parse the base config (handles num_iq_samples properly)
    base_config = parse_base_config(parser)

    # Get the args again to extract training-specific fields
    args = parser.parse_args()

    # Create TrainingConfig by combining base config and training args
    training_config = TrainingConfig(
        **vars(base_config),

        # Add training fields
        online_linear_eval=args.online_linear_eval,
        ssl_model=args.ssl_model,
        training_path=args.training_path,
        num_epochs=args.num_epochs,
        two_tower_num_epochs=args.two_tower_num_epochs,
        training_stage=args.training_stage,
    )

    return training_config
