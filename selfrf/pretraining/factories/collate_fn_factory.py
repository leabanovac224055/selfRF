import torch
from typing import Any, Callable, List, Tuple, Union

from selfrf.pretraining.utils.enums import CollateType, DatasetType
from selfrf.pretraining.config import TrainingConfig, EvaluationConfig
from selfrf.pretraining.config.training_config import TrainingStage


def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig], is_val=False) -> Callable:
    """
    Build collate function based on the SSL model type.
    """
    
    '''if getattr(config, "mode", None) == "metadata":
        # Validaiton mode = true
        if isinstance(config, EvaluationConfig):
            print("[DEBUG] Using simple_metadata_collate_fn_eval for metadata mode")
            return metadata_collate_fn_eval
        else:
            # Training mode
            print("[DEBUG] Using simple_metadata_collate_fn for metadata mode")
            return metadata_collate_fn'''
    
    print(f"[DEBUG] Building collate function. Validation: {is_val} | Stage: {getattr(config, 'training_stage', 'UNKNOWN')}")

    # 1. Metadata tower training (metadata only)
    if getattr(config, "training_stage", None) == TrainingStage.METADATA:
        return metadata_collate_fn_eval if is_val else metadata_collate_fn

    # 2. Fusion model training (two-tower)
    if getattr(config, "training_stage", None) == TrainingStage.FUSION:
        return TwoTowerCollate(validation_mode=is_val)

    # 3. Evaluation mode fallback
    if isinstance(config, EvaluationConfig):
        if config.dataset == DatasetType.TWO_TOWER_NARROWBAND:
            return TwoTowerCollate(validation_mode=True)
        return collate_fn_evaluation

    # 4. SSL: use standard multi/single-view logic
    if getattr(config, "online_linear_eval", False) and is_val:
        return MultiViewCollate(validation_mode=True)

    ssl_model = config.ssl_model

    if ssl_model.collate_type == CollateType.MULTI_VIEW:
        print("[DEBUG] Using MultiViewCollate")
        return MultiViewCollate()
    elif ssl_model.collate_type == CollateType.SINGLE_VIEW:
        print("[DEBUG] Using SingleViewCollate")
        return SingleViewCollate()
    else:
        raise ValueError(f"Unknown collate type for SSL model: {ssl_model}")


def collate_fn_evaluation(batch):
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    indices = [y[0] for y in targets]
    # Convert targets to tensor
    indices = torch.tensor(indices)
    
    names = [y[1][0] if isinstance(y[1], tuple) else y[1] for y in targets]
    
    return tensors, (indices, names)


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
    
    def __init__(self, validation_mode=False):
        self.validation_mode = validation_mode    

    def __call__(self, batch: List[Any]) -> Tuple[torch.Tensor, ...]:
        # Unpacks [(views1, label1), (views2, label2), ...]
        views, targets = zip(*batch)
        # Unpacks [(view1_1, view1_2), (view2_1, view2_2), ...]
        view1s, view2s = zip(*views)
        
        targets = torch.tensor([y[0] for y in targets]).flatten()

        if self.validation_mode:
            stacked_views = torch.stack(view1s)
            if stacked_views.ndim == 2:
                stacked_views = stacked_views.view(stacked_views.shape[0], 2, -1)
            return stacked_views, targets
        else:
            return (
                torch.stack(view1s),
                torch.stack(view2s)
            ), targets


class TwoTowerCollate:
    def __init__(self, validation_mode=False):
        self.validation_mode = validation_mode
        print(f"[DEBUG] TwoTowerCollate initialized. Validation mode: {self.validation_mode}")
        
    def __call__(self, batch: List[Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, list]]]:
        """
        Collate function for two-tower datasets.
        Training: ((view1, view2), metadata_vector, label)
        Evaluation: ((view1, view2), metadata_vector, (label_batch, name_batch))
        """
        
        views, metadata_vector, targets = zip(*batch)
        view1s, view2s = zip(*views)
        
        if self.validation_mode:
            # Stack views correctly to form [B, C, L] shape
            view1_batch = torch.stack(view1s)  # [B, 4096]
            
            # Ensure the view has the correct number of channels (2)
            if view1_batch.ndim == 2:
                view1_batch = view1_batch.unsqueeze(1)  # Add channel dimension
                view1_batch = torch.cat([view1_batch, view1_batch], dim=1)  # Duplicate to create two channels

            # Separate indices and names from targets
            indices = [y[0] for y in targets]
            names = [y[1] if isinstance(y, tuple) else "unknown" for y in targets]
            indices = torch.tensor(indices)
            return view1_batch, (indices, names)
        else:
            # In training mode, use both views
            view1_batch = torch.stack(view1s)
            view2_batch = torch.stack(view2s)
            metadata_batch = torch.stack(metadata_vector)
            label_batch = torch.tensor(targets).long()

            return (view1_batch, view2_batch), metadata_batch, label_batch

def metadata_collate_fn(batch):
    metadata = []
    targets = []

    for item in batch:
        # Extract metadata, which is the first element and contains a list of tensors
        if isinstance(item[0], list):
            # Concatenate metadata tensors along the last dimension
            meta_tensor = torch.cat(item[0], dim=-1)
            # Flatten the metadata tensor
            meta_tensor = meta_tensor.view(-1)
            metadata.append(meta_tensor)
        else:
            metadata.append(item[0].detach().clone())

        # Extract target
        try:
            # Check if the target is already an integer
            if isinstance(item[1], int):
                target = item[1]
            # Check if the target is a tuple of length 2, with an integer inside
            elif isinstance(item[1], tuple) and isinstance(item[1][0], int):
                target = item[1][0]
            # Handle nested tuple structure
            elif isinstance(item[1], tuple) and isinstance(item[1][0], (list, tuple)):
                target = item[1][0][0]
            else:
                raise ValueError("Unexpected target format")
            targets.append(target)
        except (IndexError, TypeError, ValueError) as e:
            print(f"[ERROR] Issue extracting target from item: {item}, Error: {e}")
            continue

    # Stack metadata and targets into tensors
    metadata = torch.stack(metadata)
    targets = torch.tensor(targets, dtype=torch.long)

    return metadata, targets

def metadata_collate_fn_eval(batch):
    metadata = []
    indices = []
    names = []

    for item in batch:
        # Extract metadata (first element) and label (second element)
        meta_tensor = item[0]
        label = item[1]

        # Add to the respective lists
        metadata.append(meta_tensor)

        # Handling the label, which might be a tuple during evaluation
        if isinstance(label, tuple):
            indices.append(label[0])
            names.append(label[1])
        else:
            indices.append(label)
            names.append("unknown")

    # Stack metadata into a tensor
    metadata = torch.stack(metadata)
    indices = torch.tensor(indices)

    return metadata, (indices, names)