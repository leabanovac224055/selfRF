import torch
from typing import Any, Callable, List, Tuple, Union

from selfrf.pretraining.utils.enums import CollateType
from selfrf.pretraining.config import TrainingConfig, EvaluationConfig


def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig]) -> Callable:
    """
    Build collate function based on the SSL model type.
    """

    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation
    ssl_model = config.ssl_model

    if ssl_model.collate_type == CollateType.MULTI_VIEW:
        return MultiViewCollate()
    elif ssl_model.collate_type == CollateType.SINGLE_VIEW:
        return SingleViewCollate()
    else:
        raise ValueError(f"Unknown collate type for SSL model: {ssl_model}")


def collate_fn_evaluation(batch):
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return tensors, targets


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
