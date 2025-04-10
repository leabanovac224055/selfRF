import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatMLPHead(nn.Module):
    def __init__(
        self,
        iq_embedding_dim: int,
        meta_embedding_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = iq_embedding_dim + meta_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, iq_embedding: torch.Tensor, meta_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq_embedding: Tensor of shape (B, D_iq)
            meta_embedding: Tensor of shape (B, D_meta)
        Returns:
            fused embedding: Tensor of shape (B, output_dim)
        """
        x = torch.cat([iq_embedding, meta_embedding], dim=1)
        return self.mlp(x)


if __name__ == "__main__":
    iq = torch.randn(4, 256)          # B, D_iq
    meta = torch.randn(4, 64)         # B, D_meta
    model = ConcatMLPHead(256, 64)
    fused = model(iq, meta)
    print("Fused shape:", fused.shape)  # (4, 128)
