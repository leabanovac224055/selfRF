import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """Computes the NT-Xent loss for positive pairs (z_i, z_j)"""
        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings to form positive pairs
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # Remove self-similarity
        batch_size = z_i.shape[0]
        mask = torch.eye(2 * batch_size, device=similarity_matrix.device).bool()

        # Scale by temperature
        logits = similarity_matrix / self.temperature
        logits = logits[~mask].view(2 * batch_size, -1)

        # Create positive pair labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z_i.device)

        loss = self.criterion(logits, labels)
        return loss

class MetadataTrainModule(LightningModule):
    def __init__(self, metadata_tower, lr=1e-3, temperature=0.5):
        super().__init__()
        self.metadata_tower = metadata_tower
        self.loss_fn = NTXentLoss(temperature=temperature)  # Use NT-Xent loss
        self.lr = lr

    def forward(self, x):
        return self.metadata_tower(x)

    def training_step(self, batch, batch_idx):
        metadata, _ = batch
        z_i = self(metadata)
        z_j = self(metadata)  # Simulate a second view of the same data
        loss = self.loss_fn(z_i, z_j)  # NT-Xent loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metadata, _ = batch
        z_i = self(metadata)
        z_j = self(metadata)  # Simulate a second view of the same data
        loss = self.loss_fn(z_i, z_j)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.metadata_tower.parameters(), lr=self.lr)
        return optimizer
