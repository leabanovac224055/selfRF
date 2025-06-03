import argparse
from dataclasses import dataclass

from .base_config import BaseConfig, add_base_config_args, parse_base_config

DEFAULT_TSNE = True
DEFAULT_KNN = True
DEFAULT_N_NEIGHBORS = 50
DEFAULT_EVALUATION_PATH = './evaluation'

@dataclass
class EvaluationConfig(BaseConfig):
    model_path: str = '.'
    tsne: bool = DEFAULT_TSNE
    knn: bool = DEFAULT_KNN
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    evaluation_path: str = DEFAULT_EVALUATION_PATH


def add_evaluation_config_args(parser: argparse.ArgumentParser) -> None:
    add_base_config_args(parser)
    parser.add_argument(
        '--model-path',
        type=str,
        default='.',
    )
    parser.add_argument(
        '--tsne',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_TSNE,
    )
    parser.add_argument(
        '--knn',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_KNN,
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=DEFAULT_N_NEIGHBORS,
    )
    parser.add_argument(
        '--evaluation-path',
        type=str,
        default=DEFAULT_EVALUATION_PATH,
    )

def parse_evaluation_config() -> EvaluationConfig:
    """Parse command line arguments into an EvaluationConfig object.

    Creates a base config first, then builds evaluation config from it.
    """
    # Create parser with description
    parser = argparse.ArgumentParser(description="Evaluation Config")
    add_evaluation_config_args(parser)

    # First parse the base config (handles num_iq_samples properly)
    base_config = parse_base_config(parser)

    # Get the args again to extract evaluation-specific fields
    args = parser.parse_args()

    # Create EvaluationConfig by combining base config and evaluation args
    evaluation_config = EvaluationConfig(
        **vars(base_config),  # Unpack base config

        # Add evaluation fields
        model_path=args.model_path,
        tsne=args.tsne,
        knn=args.knn,
        n_neighbors=args.n_neighbors,
        evaluation_path=args.evaluation_path
    )

    return evaluation_config