from typing import Union

import torch

from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.config.training_config import TrainingConfig


def collate_fn(batch):
    # Unpacks [(views1, label1), (views2, label2), ...]
    views, targets = zip(*batch)
    # Unpacks [(view1_1, view1_2), (view2_1, view2_2), ...]
    view1s, view2s = zip(*views)

    return (
        torch.stack(view1s),  # Batch of first views
        torch.stack(view2s)   # Batch of second views
    ), torch.tensor(targets)


def collate_fn_evaluation(batch):
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return tensors, targets


def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig]):
    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation
    else:
        return collate_fn
