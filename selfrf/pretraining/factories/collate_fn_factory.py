from typing import Union
import torch
from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.config.training_config import TrainingConfig


def collate_fn(batch):
    views, targets = zip(*batch)
    view1s, view2s = zip(*views)

    print(f"🔍 Batch Targets Before Processing: {targets}")  # Debugging line

    extracted_targets = []
    for t in targets:
        if isinstance(t, dict):
            # ✅ Extract integer value
            extracted_targets.append(t.get("class_index", 0))
        else:
            extracted_targets.append(t)  # Already a number

    print(f"✅ Processed Targets: {extracted_targets}")  # Debugging line

    return (
        torch.stack(view1s),
        torch.stack(view2s)
    ), torch.tensor(extracted_targets, dtype=torch.long)


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
