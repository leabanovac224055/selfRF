import argparse
from dataclasses import dataclass

from selfrf.pretraining.utils.enums import SSLModelType
from .base_config import BaseConfig, add_base_config_args, parse_base_config

DEFUALT_ONLINE_LINEAR_EVAL = False
DEFAULT_SSL_MODEL = SSLModelType.BYOL
DEFAULT_TRAINING_PATH = './train'
DEFAULT_NUM_EPOCHS = 100


@dataclass
class TrainingConfig(BaseConfig):
    online_linear_eval: bool = DEFUALT_ONLINE_LINEAR_EVAL
    ssl_model: SSLModelType = DEFAULT_SSL_MODEL
    training_path: str = DEFAULT_TRAINING_PATH
    num_epochs: int = DEFAULT_NUM_EPOCHS


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
    )

    return training_config
