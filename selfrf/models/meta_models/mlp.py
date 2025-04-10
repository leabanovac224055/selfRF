import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A simple MLP for processing metadata feature vectors like:
    [center_frequency, bandwidth, duration, ...]
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Metadata input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, output_dim)
        """
        return self.net(x)
