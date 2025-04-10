import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class TwoTowerTrainModule(LightningModule):
    def __init__(
        self,
        iq_encoder: nn.Module,
        metadata_tower: nn.Module,
        fusion_head: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.iq_encoder = iq_encoder
        self.metadata_tower = metadata_tower
        self.fusion_head = fusion_head
        self.loss_fn = loss_fn
        self.lr = lr

        # Freeze IQ encoder
        for param in self.iq_encoder.parameters():
            param.requires_grad = False
        self.iq_encoder.eval()

    def forward(self, x_iq, x_meta):
        # x_iq: shape [B, 2, 4096]
        # x_meta: shape [B, D]

        iq_emb = self.iq_encoder(x_iq)
        meta_emb = self.metadata_tower(x_meta)
        fused = self.fusion_head(iq_emb, meta_emb)

        return fused

    def training_step(self, batch, batch_idx):
        (view1, view2), metadata, _ = batch

        # Get embeddings (no gradients from iq_encoder)
        with torch.no_grad():
            iq_emb1 = self.iq_encoder(view1)
            iq_emb2 = self.iq_encoder(view2)

        # Train metadata tower + fusion
        meta_emb = self.metadata_tower(metadata)

        fused1 = self.fusion_head(iq_emb1, meta_emb)
        fused2 = self.fusion_head(iq_emb2, meta_emb)

        loss = self.loss_fn(fused1, fused2)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.metadata_tower.parameters()) + \
            list(self.fusion_head.parameters())
        return torch.optim.Adam(params, lr=self.lr)
