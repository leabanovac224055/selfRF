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
        '--ssl-model',
        type=lambda x: SSLModelType(x),
        choices=list(SSLModelType),
        default=DEFAULT_SSL_MODEL
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

    # Add XCiT-specific parameters
    parser.add_argument(
        '--drop-path-rate',
        type=float,
        default=BaseConfig.drop_path_rate
    )
    parser.add_argument(
        '--drop-rate',
        type=float,
        default=BaseConfig.drop_rate
    )
    parser.add_argument(
        '--ds-method',
        type=str,
        default=BaseConfig.ds_method
    )
    parser.add_argument(
        '--ds-rate',
        type=int,
        default=BaseConfig.ds_rate
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
        **{k: v for k, v in vars(base_config).items() if k not in ["drop_path_rate", "drop_rate", "ds_method", "ds_rate"]},

        # Add training fields
        online_linear_eval=args.online_linear_eval,
        ssl_model=args.ssl_model,
        training_path=args.training_path,
        num_epochs=args.num_epochs,
        drop_path_rate=args.drop_path_rate,
        drop_rate=args.drop_rate,
        ds_method=args.ds_method,
        ds_rate=args.ds_rate,
    )

    return training_config
