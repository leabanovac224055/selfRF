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
        lr: float = 1e-3,
        temperature: float = 0.1,
        freeze_iq_encoder: bool = True,
        freeze_metadata_tower: bool = True
    ):
        super().__init__()
        self.iq_encoder = iq_encoder
        self.metadata_tower = metadata_tower
        self.fusion_head = fusion_head
        self.temperature = temperature
        self.lr = lr

        # Optionally freeze IQ encoder
        if freeze_iq_encoder:
            for param in self.iq_encoder.parameters():
                param.requires_grad = False
            self.iq_encoder.eval()

        # Optionally freeze metadata tower
        if freeze_metadata_tower:
            for param in self.metadata_tower.parameters():
                param.requires_grad = False
            self.metadata_tower.eval()

    def forward(self, x_iq, x_meta):
        # x_iq: shape [B, 2, 4096]
        # x_meta: shape [B, D]

        iq_emb = self.iq_encoder(x_iq)
        meta_emb = self.metadata_tower(x_meta)
        fused = self.fusion_head(iq_emb, meta_emb)

        return fused
    
    def nt_xent_loss(self, z1, z2):
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        N = z1.size(0)
        labels = torch.arange(N, device=self.device)
        labels = torch.cat([labels, labels], dim=0)

        # Mask self-similarity
        mask = ~torch.eye(2 * N, dtype=torch.bool, device=self.device)

        sim = sim[mask].view(2 * N, -1)
        positives = F.cosine_similarity(z1, z2).repeat(2)

        logits = torch.cat([positives.unsqueeze(1), sim], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * N, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        (view1, view2), metadata, _ = batch
        iq_emb1 = self.iq_encoder(view1)
        iq_emb2 = self.iq_encoder(view2)
        meta_emb = self.metadata_tower(metadata)
        fused1 = self.fusion_head(iq_emb1, meta_emb)
        fused2 = self.fusion_head(iq_emb2, meta_emb)
        loss = self.nt_xent_loss(fused1, fused2)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.fusion_head.parameters())
        if any(p.requires_grad for p in self.iq_encoder.parameters()):
            params += list(self.iq_encoder.parameters())
        if any(p.requires_grad for p in self.metadata_tower.parameters()):
            params += list(self.metadata_tower.parameters())

        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer
    