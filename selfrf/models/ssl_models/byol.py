from typing import List, Tuple
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
import copy

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.utils.lars import LARS


class BYOL(LightningModule):
    def __init__(
        self,
        num_classes: int,
        batch_size_per_device: int,
        backbone: nn.Module,
        num_ftrs: int = 2048,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        start_momentum: float = 0.97,
        use_online_linear_eval: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['backbone'])
        self.batch_size_per_device = batch_size_per_device

        self.backbone = backbone

        self.projection_head = BYOLProjectionHead(
            num_ftrs, hidden_dim, out_dim)
        self.prediction_head = BYOLPredictionHead(out_dim, hidden_dim, out_dim)

        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = copy.deepcopy(self.projection_head)

        self.criterion = NegativeCosineSimilarity()
        self.start_momentum = start_momentum

        self.use_online_linear_eval = use_online_linear_eval

        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(
                num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, tuple):
            print(f"[BYOL DEBUG] ❌ Received tuple instead of Tensor: {type(x)}")
            print(f"Contents: {[type(e) for e in x]}")
            raise TypeError("Expected Tensor input, got tuple. Likely collate_fn issue.")
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)

        projections = self.projection_head(features)
        predictions = self.prediction_head(projections)
        return features, predictions

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_backbone(x)
        projections = self.teacher_projection_head(features)
        return projections

    def training_step(self, batch: Tuple[List[Tensor], Tensor, List[str]],
                      batch_idx: int) -> Tensor:

        # Momentum update teacher.
        # Settings follow original code for 100 epochs which are slightly different
        # from the paper, see:
        # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.start_momentum,
            end_value=1.0,
        )
        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head,
                        self.teacher_projection_head, m=momentum)

        # get views and targets from batch
        if isinstance(batch, tuple) and len(batch) == 3:
            views, _, targets = batch  # ignore metadata
        else:
            views, targets = batch

        x0 = views[0]
        x1 = views[1]

        # forward pass and loss calculation
        teacher_projections_0 = self.forward_teacher(x0)
        teacher_projections_1 = self.forward_teacher(x1)

        student_features_0, student_predictions_0 = self.forward_student(x0)
        _, student_predictions_1 = self.forward_student(x1)

        # NOTE: Factor 2 because: L2(norm(x), norm(y)) = 2 - 2 * cossim(x, y)
        loss_0 = 2 * self.criterion(teacher_projections_0,
                                    student_predictions_1)
        loss_1 = 2 * self.criterion(teacher_projections_1,
                                    student_predictions_0)
        # NOTE: No mean because original code only takes mean over batch dimension, not
        # views.
        loss = loss_0 + loss_1
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=len(targets)
        )

        if not self.use_online_linear_eval:
            return loss

        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (student_features_0.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, batch_size=len(targets))

        total_loss = loss + cls_loss
        self.log("total_loss", total_loss,
                 prog_bar=True, batch_size=len(targets))
        return total_loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        if not self.use_online_linear_eval:
            return

        # Unpack batch depending on structure
        if isinstance(batch, tuple) and len(batch) == 3:
            x0, _, targets = batch  # x0 = view1 batch, second element is metadata, ignore it
        else:
            x0, targets = batch  # for IQDMNarrowband or classic BYOL

        # Forward only x0 through backbone
        features = self.forward(x0).flatten(start_dim=1)

        # Online linear evaluation
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss


    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.projection_head,
                self.prediction_head,
            ]
        )

        p = [
            {"name": "byol", "params": params},
            {
                "name": "byol_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },

        ]
        if self.use_online_linear_eval:
            p.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            )

        optimizer = LARS(
            p,
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.45 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        estimated_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(estimated_steps / self.trainer.max_epochs * 10),
                max_epochs=estimated_steps,
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx=None, metric=None):
        # ⚠️ Clamp step to avoid crash at the final iteration
        try:
            scheduler.step()
        except ValueError as e:
            if "Total step number 0" in str(e):
                self.print("⚠️ Skipping final lr_scheduler step due to zero steps.")
            else:
                raise e

