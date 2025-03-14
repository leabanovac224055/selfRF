import torch
from typing import Any, Callable, List, Tuple

from selfrf.pretraining.utils.enums import SSLModelType
from selfrf.pretraining.config import TrainingConfig


def build_collate_fn(config: TrainingConfig) -> Callable:
    """
    Build collate function based on the SSL method and number of views.

    Args:
        config: Configuration object with ssl_method attribute

    Returns:
        Appropriate collate function
    """

    # Build appropriate collate function
    # if config.ssl_model == SSLModelType.MAE:
    #    return SingleViewCollate()
    if config.ssl_model == SSLModelType.BYOL or config.ssl_model == SSLModelType.DINO:
        return MultiViewCollate()
    else:
        raise ValueError(
            f"Unknown SSL method: {config.ssl_model}.")


class SingleViewCollate:
    """Collate function for single-view SSL methods like MAE."""

    def __call__(self, batch: List[Any]) -> torch.Tensor:
        """
        Convert a batch of single-view samples into a batched tensor.

        Each sample is expected to be either:
        - A tensor directly
        - A list/tuple containing a single tensor
        """
        datas, targets = zip(*batch)

        # Stack tensors
        return torch.stack(datas), torch.tensor(targets)


class MultiViewCollate:
    """Collate function for multi-view SSL methods like BYOL or DINO."""

    def __call__(self, batch: List[Any]) -> Tuple[torch.Tensor, ...]:
        # Unpacks [(views1, label1), (views2, label2), ...]
        views, targets = zip(*batch)
        # Unpacks [(view1_1, view1_2), (view2_1, view2_2), ...]
        view1s, view2s = zip(*views)
        return (
            torch.stack(view1s),  # Batch of first views
            torch.stack(view2s)   # Batch of second views
        ), torch.tensor(targets)


""" def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig]):
    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation
 """
