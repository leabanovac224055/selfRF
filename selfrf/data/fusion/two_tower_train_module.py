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

    @property
    def backbone(self):
        return self.iq_encoder

    def forward(self, x_iq, x_meta):
        iq_emb = self.iq_encoder(x_iq)
        
        # Unpack if iq_encoder returns a tuple (cls_token, pooled_token)
        if isinstance(iq_emb, tuple):
            iq_emb = iq_emb[0]  # or iq_emb[0] depending on your encoder architecture
        
        meta_emb = self.metadata_tower(x_meta)
        fused = self.fusion_head(iq_emb, meta_emb)
        return fused

    def nt_xent_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        N = z1.size(0)
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        sim_matrix = sim_matrix / self.temperature

        positives = torch.cat([
            torch.diag(sim_matrix, N),
            torch.diag(sim_matrix, -N)
        ], dim=0)

        logits = torch.cat([positives.unsqueeze(1), sim_matrix], dim=1)
        labels = torch.zeros(2 * N, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        # Unpack batch with masks
        ((view1_iq, view1_mask), (view2_iq, view2_mask)), metadata, _ = batch

        # NaN checks for inputs
        for tensor, name in [
            (view1_iq, "view1_iq"), (view1_mask, "view1_mask"),
            (view2_iq, "view2_iq"), (view2_mask, "view2_mask"),
            (metadata, "metadata")
        ]:
            if torch.isnan(tensor).any():
                raise ValueError(f"NaNs in {name}")

        # Forward pass for IQ encoder
        iq_emb1_tuple = self.iq_encoder(view1_iq)
        iq_emb2_tuple = self.iq_encoder(view2_iq)

        # Always unpack (cls_token, pooled_token)
        iq_emb1_cls, iq_emb1_pool = iq_emb1_tuple
        iq_emb2_cls, iq_emb2_pool = iq_emb2_tuple

        # Use cls_token as the representation
        iq_emb1 = iq_emb1_cls
        iq_emb2 = iq_emb2_cls

        # Metadata tower
        meta_emb = self.metadata_tower(metadata)

        # Sanity checks for embeddings
        for name, tensor in {
            "iq_emb1": iq_emb1, "iq_emb2": iq_emb2, "meta_emb": meta_emb
        }.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"NaNs in {name}")
            if torch.all(tensor == 0):
                raise ValueError(f"All-zero values in {name}")

        fused1 = self.fusion_head(iq_emb1, meta_emb)
        fused2 = self.fusion_head(iq_emb2, meta_emb)

        for name, tensor in {
            "fused1": fused1, "fused2": fused2
        }.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"NaNs in {name}")
            if torch.all(tensor == 0):
                raise ValueError(f"All-zero values in {name}")

        loss = self.nt_xent_loss(fused1, fused2)

        if not torch.isfinite(loss):
            self.print("⚠️ NaN or Inf loss encountered — skipping batch")
            return None

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = list(self.fusion_head.parameters())
        if any(p.requires_grad for p in self.iq_encoder.parameters()):
            params += list(self.iq_encoder.parameters())
        if any(p.requires_grad for p in self.metadata_tower.parameters()):
            params += list(self.metadata_tower.parameters())

        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer