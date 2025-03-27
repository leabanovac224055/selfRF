from typing import Literal
import timm
import torch
from torch import nn
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

from selfrf.pretraining.utils.enums import BackboneProvider

__all__ = ["build_resnet2d"]


class Detectron2ResNet(nn.Module):
    """ResNet backbone from Detectron2 with feature extraction capabilities."""

    def __init__(
        self,
        resnet: nn.Module,
        n_features,
        stage_name="res5",
        feature_only: bool = False,
    ):
        super().__init__()
        self.resnet = resnet
        self.stage = stage_name
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.feature_only = feature_only  # Flag to determine output type

    def forward(self, x):
        features = self.resnet(x)

        # Extract the feature map from the specified stage
        stage_output = features[self.stage]

        if self.feature_only:
            # Return feature maps
            return stage_output
        else:
            # Return vector
            return self.pool(stage_output)


def build_detectron2_resnet(
    input_channels: int,
    n_features: int,
    version: str = "50",
    feature_only: bool = True,
) -> nn.Module:
    """Build ResNet backbone using Detectron2."""

    # Initialize config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        f"COCO-Detection/faster_rcnn_R_{version}_FPN_3x.yaml"
    ))

    # Configure model
    cfg.MODEL.WEIGHTS = ""  # No pretrained weights
    cfg.MODEL.PIXEL_MEAN = [0] * input_channels  # Zero mean per channel
    cfg.MODEL.PIXEL_STD = [1.0] * input_channels   # Unit std per channel
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    det_model = build_model(cfg)

    return Detectron2ResNet(
        resnet=det_model.backbone.bottom_up,
        stage_name="res5",
        n_features=n_features,
        feature_only=feature_only,
    )


def build_resnet2d(
    input_channels: int,
    n_features: int = 2048,
    version: str = "50",
    provider: BackboneProvider = BackboneProvider.TIMM,
    feature_only: bool = False,
):
    """Constructs and returns a version of the ResNet model.
    Args:

        input_channels (int):
            Number of input channels; should be 2 for complex spectrograms

        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        version (str):
            Specifies the version of resnet to use, e.g., '18', '34' or '50'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """

    if provider is BackboneProvider.TIMM:
        model = timm.create_model(
            "resnet" + version,
            in_chans=input_channels,
            features_only=feature_only,
        )

        model.fc = nn.Linear(model.fc.in_features, n_features)
        return model

    elif provider is BackboneProvider.DETECTRON2:

        return build_detectron2_resnet(
            input_channels=input_channels,
            n_features=n_features,
            version=version,
            features_only=feature_only,
        )

    else:
        raise ValueError(f"{provider} does not provider a ResNet 2D backbone.")
