from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
import torch

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from selfrf.finetuning.detection.detectron2.config import Detectron2Config


def build_vitdet_b_model_config():
    """Build model configuration."""
    model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

    model.backbone.net.img_size = 512
    model.backbone.square_pad = 512
    model.backbone.net.in_chans = 1  # Single channel input

    # Disable mask prediction in ROI heads
    model.roi_heads.mask_in_features = None
    model.roi_heads.mask_pooler = None
    model.roi_heads.mask_head = None

    model.pixel_mean = [0.0]  # Single channel mean
    model.pixel_std = [1.0]   # Single channel std
    model.input_format = "L"  # Grayscale format

    return model


def build_vitdet_b_training_config(config: Detectron2Config):
    """Build training configuration."""
    train = model_zoo.get_config("common/train.py").train
    train.amp.enabled = torch.cuda.is_available()
    train.ddp.fp16_compression = True
    train.max_iter = config.max_iter
    train.eval_period = 1000
    return train


def build_vitdet_b_lr_multiplier_config(config: Detectron2Config):
    """Build learning rate multiplier configuration.

    Custom learning rate schedule for RF COCO dataset:
    - Warmup for first 1000 iterations
    - Drop LR at 80% and 90% of training
    """
   # Determine if we're in testing mode (very small max_iter)
    testing_mode = config.max_iter < 100

    # Calculate milestones based on percentages
    milestones = [
        int(0.8 * config.max_iter),  # Drop LR at 80% of training
        int(0.9 * config.max_iter),  # Drop LR at 90% of training
    ]

    # Adjust warmup length appropriately for testing
    if testing_mode:
        # For testing, use just 20% of iterations for warmup
        warmup_length = 0.2
        warmup_iters = int(config.max_iter * 0.2)
    else:
        # For production, use the standard 1000 iterations
        # Cap at 90% to be safe
        warmup_length = min(1000 / config.max_iter, 0.9)
        warmup_iters = 1000

    print(f"LR Schedule: max_iter={config.max_iter}, milestones={milestones}, "
          f"warmup_length={warmup_length}, warmup_iters={warmup_iters}")

    return L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            # Normalize to [0,1]
            milestones=[m/config.max_iter for m in milestones],
            num_updates=config.max_iter,
        ),
        warmup_length=warmup_length,  # Now properly scaled
        warmup_factor=0.001,
    )


def build_vitdet_b_optimizer_config():
    """Build optimizer configuration."""
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    optimizer.params.lr_factor_func = partial(
        get_vit_lr_decay_rate,
        num_layers=12,
        lr_decay_rate=0.7
    )
    optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
    return optimizer
