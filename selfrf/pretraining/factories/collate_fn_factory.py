from typing import Union

import torch

from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.config.training_config import TrainingConfig


def collate_fn(batch):
    views, targets = zip(*batch)

    # Compute the number of views for each sample
    view_counts = [len(v) for v in views]
    print(f"Views per sample before stacking: {view_counts}")

    # Dynamically determine the minimum number of views available in the batch
    num_views = min(view_counts)

    print(f"Batch will be processed with {num_views} views per sample.")

    # Stack only the available views
    stacked_views = [torch.stack([v[i] for v in views if len(v) > i])
                     for i in range(num_views)]

    print(f"Final stacked views count: {len(stacked_views)}")
    print(f"Stacked view shapes: {[v.shape for v in stacked_views]}")

    # **Instead of padding, pick only the first target from each sample**
    selected_targets = [t[0] if isinstance(t, list) and len(t) > 0 else [
        0, 0, 0, 0] for t in targets]

    targets_tensor = torch.tensor(selected_targets, dtype=torch.float32)

    print(f"Final Targets Shape: {targets_tensor.shape}")

    return stacked_views, targets_tensor


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
