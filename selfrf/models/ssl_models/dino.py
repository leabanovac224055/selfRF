from typing import Tuple
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy

from lightly.loss import DINOLoss
from lightly.models.modules.heads import DINOProjectionHead as DINOHead
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.lars import LARS
from lightly.transforms import DINOTransform


class DINO(LightningModule):
    def __init__(self,
                 num_classes: int,
                 batch_size_per_device: int,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 hidden_dim: int = 4096,
                 out_dim: int = 128,  # usually 65536,but for simple training, we use 128
                 momentum_teacher: float = 0.996,
                 use_online_linear_eval: bool = False,
                 ):
        super(DINO, self).__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.batch_size_per_device = batch_size_per_device

        self.backbone = backbone
        self.projection_head = DINOHead(
            # Force 128 for simple training, because of memory issue
            num_ftrs, hidden_dim, bottleneck_dim=256, output_dim=128)
        print(
            f"Student head output shape: {self.projection_head(torch.randn(1, num_ftrs)).shape}")

        # Teacher model (Momentum updated)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = DINOHead(
            num_ftrs, hidden_dim, bottleneck_dim=256, output_dim=128)  # Match 128
        print(
            f"Teacher head output shape: {self.teacher_projection_head(torch.randn(1, num_ftrs)).shape}")

        self.criterion = DINOLoss(out_dim)
        self.momentum_teacher = momentum_teacher
        self.use_online_linear_eval = use_online_linear_eval

        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(
                num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        return projections

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_backbone(x).flatten(start_dim=1)
        projections = self.teacher_projection_head(features)
        return projections

    def measure_similarity(self, global_views: torch.Tensor, local_views: torch.Tensor):
        """Computes similarity between teacher (global) and student (local) views."""
        similarities = []
        mse_values = []

        for i in range(global_views.shape[0]):  # Iterate over batch
            sim = F.cosine_similarity(
                global_views[i].flatten(), local_views[i].flatten(), dim=0).item()
            mse = F.mse_loss(global_views[i], local_views[i]).item()
            similarities.append(sim)
            mse_values.append(mse)

        return torch.tensor(similarities).mean().item(), torch.tensor(mse_values).mean().item()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        views, _ = batch
        x0, x1 = views[0], views[1]

        student_proj_0 = self.forward_student(x0)
        student_proj_1 = self.forward_student(x1)

        with torch.no_grad():
            teacher_proj_0 = self.forward_teacher(x0)
            teacher_proj_1 = self.forward_teacher(x1)

        # Compute similarity between teacher & student
        similarity, _ = self.measure_similarity(teacher_proj_0, student_proj_0)

        # **Fix:** Pass similarity only when needed
        if hasattr(DINOTransform, "similarity"):
            dino_transform = DINOTransform(similarity=similarity)
        else:
            dino_transform = DINOTransform()

        loss = self.criterion(
            teacher_out=[teacher_proj_0, teacher_proj_1],
            student_out=[student_proj_0, student_proj_1],
            epoch=self.current_epoch
        )

        self.log("similarity", similarity, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch))

        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + \
            list(self.projection_head.parameters())
        optimizer = LARS(params, lr=0.3, momentum=0.9, weight_decay=1.5e-6)
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer, warmup_epochs=10, max_epochs=self.trainer.max_epochs
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
