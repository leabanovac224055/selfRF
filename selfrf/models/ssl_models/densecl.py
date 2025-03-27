import math
import copy
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from lightly.loss import NTXentLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum, select_most_similar
from lightly.models.modules import DenseCLProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.utils.benchmarking import OnlineLinearClassifier


class DenseCL(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        batch_size_per_device: int,
        start_momentum: float = 0.996,
        lambda_weight: float = 0.7,
        num_classes: int = 10,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        use_online_linear_eval: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.batch_size_per_device = batch_size_per_device

        self.backbone = backbone
        self.projection_head_global = DenseCLProjectionHead(
            input_dim, hidden_dim, output_dim)
        self.projection_head_local = DenseCLProjectionHead(
            input_dim, hidden_dim, output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(
            self.projection_head_global
        )
        self.projection_head_local_momentum = copy.deepcopy(
            self.projection_head_local)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_global_momentum)
        deactivate_requires_grad(self.projection_head_local_momentum)

        self.criterion_global = NTXentLoss(memory_bank_size=(4096, 128))
        self.criterion_local = NTXentLoss(memory_bank_size=(4096, 128))

        self.start_momentum = start_momentum
        self.lambda_weight = lambda_weight

        self.use_online_linear_eval = use_online_linear_eval

        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(
                feature_dim=output_dim,
                num_classes=num_classes)

    def forward(self, x):
        query_features = self.backbone(x)
        query_global = self.pool(query_features).flatten(start_dim=1)
        query_global = self.projection_head_global(query_global)
        query_features = query_features.flatten(start_dim=2).permute(0, 2, 1)
        query_local = self.projection_head_local(query_features)
        # Shapes: (B, H*W, C), (B, D), (B, H*W, D)
        return query_features, query_global, query_local

    @torch.no_grad()
    def forward_momentum(self, x):
        key_features = self.backbone_momentum(x)
        key_global = self.pool(key_features).flatten(start_dim=1)
        key_global = self.projection_head_global_momentum(key_global)
        key_features = key_features.flatten(start_dim=2).permute(0, 2, 1)
        key_local = self.projection_head_local_momentum(key_features)
        return key_features, key_global, key_local

    def training_step(
        self,
        batch: Tuple[List[Tensor], Tensor, List[str]],
        batch_idx: int
    ) -> Tensor:

        momentum = cosine_schedule(
            self.current_epoch,
            self.trainer.estimated_stepping_batches,
            self.start_momentum,
            1,
        )
        update_momentum(
            self.backbone,
            self.backbone_momentum,
            m=momentum,
        )
        update_momentum(
            self.projection_head_global,
            self.projection_head_global_momentum,
            m=momentum,
        )
        update_momentum(
            self.projection_head_local,
            self.projection_head_local_momentum,
            m=momentum,
        )

        x_query, x_key = batch[0]
        query_features, query_global, query_local = self(x_query)
        key_features, key_global, key_local = self.forward_momentum(x_key)

        key_local = select_most_similar(
            query_features, key_features, key_local)
        query_local = query_local.flatten(end_dim=1)
        key_local = key_local.flatten(end_dim=1)

        loss_global = self.criterion_global(query_global, key_global)
        loss_local = self.criterion_local(query_local, key_local)
        loss = (1 - self.lambda_weight) * loss_global + \
            self.lambda_weight * loss_local

        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch[1]))
        self.log('loss_global', loss_global)
        self.log('loss_local', loss_local)

        if not self.use_online_linear_eval:
            return loss

        targets = batch[1]  # targets are the labels of the batch
        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (query_global.detach(), targets), batch_idx
        )

        self.log_dict(cls_log, prog_bar=True, batch_size=len(targets))

        total_loss = loss + cls_loss
        self.log("total_loss", total_loss,
                 prog_bar=True, batch_size=len(targets))
        return total_loss

    def configure_optimizers(self):

        # Total number of training steps
        total_steps = self.trainer.estimated_stepping_batches

        # Collect parameters to optimize
        parameters = (
            list(self.backbone.parameters()) +
            list(self.projection_head_global.parameters()) +
            list(self.projection_head_local.parameters())
        )

        # If using online linear evaluation, add its parameters too
        if self.use_online_linear_eval:
            parameters += list(self.online_classifier.parameters())

        # Define the optimizer (SGD with momentum)
        optimizer = torch.optim.SGD(
            parameters,
            lr=0.03,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Define the cosine annealing learning rate scheduler
        def lr_lambda(current_step):
            return 0.5 * (1 + math.cos(current_step * math.pi / total_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Return optimizer and scheduler in PyTorch Lightning format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
