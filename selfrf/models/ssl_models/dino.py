from typing import Tuple
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
import copy
import numpy as np

from lightly.loss import DINOLoss
from lightly.models.modules.heads import DINOProjectionHead as DINOHead
from lightly.utils.benchmarking import OnlineLinearClassifier


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DINO(LightningModule):
    def __init__(self,
                 num_classes: int,
                 batch_size_per_device: int,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 hidden_dim: int = 4096,
                 out_dim: int = 128,
                 momentum_teacher: float = 0.996,
                 use_online_linear_eval: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.batch_size_per_device = batch_size_per_device
        self.backbone = backbone

        self.projection_head = DINOHead(num_ftrs, hidden_dim, bottleneck_dim=256, output_dim=out_dim)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = DINOHead(num_ftrs, hidden_dim, bottleneck_dim=256, output_dim=out_dim)
        
        self.student_temp = 0.1
        self.teacher_temp_start = 0.04
        self.teacher_temp_end = 0.07
        self.teacher_temp_warmup_epochs = 30
        
        self.criterion = DINOLoss(
            output_dim=out_dim,
            warmup_teacher_temp=self.teacher_temp_start,
            teacher_temp=self.teacher_temp_end,
            warmup_teacher_temp_epochs=self.teacher_temp_warmup_epochs,
            student_temp=self.student_temp
        )

        self.momentum_teacher = momentum_teacher
        self.use_online_linear_eval = use_online_linear_eval

        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

        # LR & WD Scheduling Setup
        self.base_lr = 0.0005
        self.min_lr = 1e-6
        self.weight_decay = 0.04
        self.weight_decay_end = 0.4
        self.warmup_epochs = 10

        self.lr_schedule = None
        self.wd_schedule = None


    def on_fit_start(self):
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        world_size = self.trainer.world_size if hasattr(self.trainer, "world_size") else 1

        scaled_lr = self.base_lr * (self.batch_size_per_device * world_size) / 256

        self.lr_schedule = cosine_scheduler(
            scaled_lr, self.min_lr,
            self.trainer.max_epochs, steps_per_epoch,
            warmup_epochs=self.warmup_epochs
        )

        self.wd_schedule = cosine_scheduler(
            self.weight_decay, self.weight_decay_end,
            self.trainer.max_epochs, steps_per_epoch
        )


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        step = self.global_step

        if step < len(self.lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr_schedule[step]
                param_group['weight_decay'] = self.wd_schedule[step]

        optimizer.step(closure=optimizer_closure)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self(x).flatten(start_dim=1)
        return self.projection_head(features)

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_backbone(x).flatten(start_dim=1)
        return self.teacher_projection_head(features)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        if isinstance(batch, tuple) and len(batch) == 2:
            # Standard SSL dataset: (views, label)
            views, _ = batch
            x0, x1 = views
            metadata = None
        elif isinstance(batch, tuple) and len(batch) == 3:
            # TwoTower: ((x0, x1), metadata, label)
            (x0, x1), metadata, _ = batch
        else:
            raise ValueError(f"[DINO DEBUG] ❌ Unexpected batch structure: {type(batch)}, len: {len(batch)}")

        # Forward passes
        student_proj_0 = self.forward_student(x0)
        student_proj_1 = self.forward_student(x1)

        with torch.no_grad():
            teacher_proj_0 = self.forward_teacher(x0)
            teacher_proj_1 = self.forward_teacher(x1)

        loss = self.criterion(
            teacher_out=[teacher_proj_0, teacher_proj_1],
            student_out=[student_proj_0, student_proj_1],
            epoch=self.current_epoch
        )

        self.log("train_loss", loss, prog_bar=True, batch_size=len(x0))
        return loss


    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.base_lr,  # will be overwritten by scheduler
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay  # will be overwritten by scheduler
        )
        return optimizer
