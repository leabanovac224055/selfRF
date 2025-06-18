import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch import Tensor
from typing import Optional, Tuple, List

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import numpy as np

class XCiT1d(nn.Module):
    """A 1D implementation of the XCiT architecture.

    Args:
        input_channels (int): Number of 1D input channels.
        n_features (int): Number of output features/classes.
        xcit_version (str): Version of XCiT model to use (e.g., 'nano_12_p16_224').
        drop_path_rate (float): Drop path rate for training.
        drop_rate (float): Dropout rate for training.
        ds_method (str): Downsampling method ('downsample' or 'chunk').
        ds_rate (int): Downsampling rate (e.g., 2 for downsampling by a factor of 2).
    """
    def __init__(
        self,
        input_channels: int,
        n_features: int = 2048,
        version: str = "nano_12_p16_224",
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.3,
        ds_method: str = "downsample",
        ds_rate: int = 2,
        provider=None,                # ✅ add this
        feature_only=False 
    ):
        super().__init__()

        # Ensure the model name is correct
        model_name = f"xcit_{version}" if not version.startswith("xcit_") else version

        # Create the backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0 if feature_only else n_features, 
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

        # Number of features from the backbone
        W = self.backbone.num_features
        self.embed_dim = W 
        self.output_dim = self.embed_dim  # ✅ Needed for fusion setup


        # Include the grouper Conv1d layer
        #self.grouper = nn.Conv1d(W, n_features, kernel_size=1)

        # Replace the patch embedding with a 1D version
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(input_channels, W, ds_rate)
        elif ds_method == "chunk":
            self.backbone.patch_embed = Chunker(input_channels, W, ds_rate)
        else:
            raise ValueError(
                f"{ds_method} is not a supported downsampling method; currently 'downsample' and 'chunk' are supported"
            )

        # Replace the classifier head with an identity layer (since we use self.grouper)
        self.backbone.head = nn.Identity()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        mdl = self.backbone
        B = x.shape[0]

        # Patch embedding
        x = self.backbone.patch_embed(x)  # Shape: [B, C, L]

        # Define H and W for 1D data
        Hp, Wp = x.shape[-1], 1  # Height is sequence length, Width is 1

        # Obtain positional encoding
        pos_encoder = PositionalEncoding1D(embed_dim=x.shape[-1])
        pos_encoding = pos_encoder(x.transpose(1, 2))  # Shape: [B, L, C]

        # Add positional encoding
        x = x.transpose(1, 2) + pos_encoding  # Shape: [B, L, C]

        # Apply transformer blocks
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)
            
        # Classification token
        cls_tokens = mdl.cls_token.expand(B, -1, -1)  # Shape: [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, Hp+1, C]

        # Apply class attention blocks
        for blk in mdl.cls_attn_blocks:
            x = blk(x)

        # Layer normalization
        x = mdl.norm(x)  # Shape: [B, Hp+1, C]

        # Apply the grouper Conv1d to the classification token
        # Extract the classification token (first token)
        cls_token = x[:, 0, :]  # Shape: [B, C]

        # ✅ Masked pooled embedding output
        sequence_tokens = x[:, 1:, :]  # [B, T, C]
        sequence_tokens = sequence_tokens.permute(0, 2, 1)  # [B, C, T]

        if mask is not None:
            # Resample mask to match sequence_tokens time length
            T_feat = sequence_tokens.shape[-1]
            mask_resampled = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),  # [B, 1, T_raw]
                size=T_feat,
                mode="linear",
                align_corners=False
            ).squeeze(1).clamp(0, 1)  # [B, T_feat]

            mask_resampled = mask_resampled.unsqueeze(1)  # [B, 1, T_feat]

            sum_feat = (sequence_tokens * mask_resampled).sum(-1)
            count = mask_resampled.sum(-1).clamp(min=1e-8)
            pooled_token = sum_feat / count  # [B, C]
        else:
            pooled_token = sequence_tokens.mean(-1)

        return cls_token, pooled_token

class ConvDownSampler(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 16):
        super().__init__()
        # Use a single convolutional layer with appropriate stride
        self.conv = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=ds_rate * 2,
            stride=ds_rate,
            padding=ds_rate // 2,
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Chunker(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 16):
        super().__init__()
        self.ds_rate = ds_rate
        self.embed = nn.Conv1d(in_chans, embed_dim, kernel_size=7, padding=3)
        self.pool = nn.AvgPool1d(kernel_size=ds_rate, stride=ds_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)  # Shape: [B, embed_dim, L]
        x = self.pool(x)   # Downsample by averaging
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.size()
        position = torch.arange(L, device=x.device).unsqueeze(1)  # Shape: [L, 1]
        div_term = torch.exp(torch.arange(0, C, 2, device=x.device) * (-torch.log(torch.tensor(10000.0)) / C))
        pe = torch.zeros(L, C, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # Shape: [B, L, C]
        return pe