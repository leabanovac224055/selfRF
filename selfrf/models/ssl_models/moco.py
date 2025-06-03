import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class MoCoV3(LightningModule):
    def __init__(self, backbone: torch.nn.Module, batch_size_per_device: int, num_classes: int, 
                 dim: int = 256, mlp_dim: int = 4096, T: float = 1.0, m: float = 0.999, use_online_linear_eval: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.m = m  # momentum
        self.T = T  # temperature for contrastive loss

        # ⚙️ Use the provided backbone
        self.backbone = backbone

        # Manually set the feature dimension for XCiT
        if hasattr(backbone, "embed_dim") and "xcit" in str(type(backbone)).lower():
            feature_dim = backbone.grouper.out_channels  # After the grouper layer
        else:
            feature_dim = backbone.num_features if hasattr(backbone, "num_features") else backbone.embed_dim

        # Projection heads for both encoders
        self.projection_head = MoCoProjectionHead(feature_dim, mlp_dim, dim)
        self.key_backbone = copy.deepcopy(self.backbone)
        self.key_projection_head = MoCoProjectionHead(feature_dim, mlp_dim, dim)

        # Contrastive loss (NT-Xent)
        self.criterion = NTXentLoss(temperature=self.T)

        # Online linear evaluation (if enabled)
        self.use_online_linear_eval = use_online_linear_eval
        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_query_encoder(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        return features, projections

    @torch.no_grad()
    def forward_key_encoder(self, x: Tensor) -> Tensor:
        x, shuffle = batch_shuffle(batch=x, distributed=self.trainer.num_devices > 1)
        features = self.key_backbone(x).flatten(start_dim=1)
        projections = self.key_projection_head(features)
        features = batch_unshuffle(features, shuffle, self.trainer.num_devices > 1)
        projections = batch_unshuffle(projections, shuffle, self.trainer.num_devices > 1)
        return projections

    def training_step(self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int) -> Tensor:
        views, targets = batch[0], batch[1]

        # Encode queries
        query_features, query_projections = self.forward_query_encoder(views[1])

        # Momentum update before encoding keys
        update_momentum(self.backbone, self.key_backbone, m=self.m)
        update_momentum(self.projection_head, self.key_projection_head, m=self.m)

        # Encode keys
        key_projections = self.forward_key_encoder(views[0])

        # Contrastive loss between queries and keys
        loss = self.criterion(query_projections, key_projections)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets))
        
        if not self.use_online_linear_eval:
            return loss

        # 📝 Online linear evaluation
        cls_loss, cls_log = self.online_classifier.training_step((query_features.detach(), targets), batch_idx)
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))

        return loss + cls_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step((features.detach(), targets), batch_idx)
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        params, params_no_weight_decay = get_weight_decay_parameters([self.backbone, self.projection_head])
        optimizer = SGD(
            [
                {"name": "mocov3", "params": params},
                {"name": "mocov3_no_weight_decay", "params": params_no_weight_decay, "weight_decay": 0.0}
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        # ✅ Only add online classifier parameters if online evaluation is enabled
        if self.use_online_linear_eval:
            optimizer.add_param_group(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                }
            )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=10,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]