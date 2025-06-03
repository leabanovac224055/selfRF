import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(
        self,
        iq_backbone: nn.Module,           # Pretrained and frozen
        metadata_tower: nn.Module,        # Trained or trainable
        fusion_head: nn.Module,           # The actual head we’ll train
    ):
        super().__init__()
        self.iq_backbone = iq_backbone
        self.metadata_tower = metadata_tower
        self.fusion_head = fusion_head

        # Optional: Freeze IQ backbone
        for param in self.iq_backbone.parameters():
            param.requires_grad = False

    def forward(self, iq_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq_input: shape (B, 2, L) – IQ time domain (e.g. [B, 2, 4096])
            metadata_input: shape (B, D) – metadata features
        
        Returns:
            fused_embedding: shape (B, output_dim)
        """
        iq_features = self.iq_backbone(iq_input)
        meta_features = self.metadata_tower(metadata_input)

        fused = self.fusion_head(iq_features, meta_features)
        return fused
