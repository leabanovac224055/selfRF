import torch
import torch.nn as nn


class ResNetWrapper(nn.Module):
    """
    Wrapper to make ResNet compatible with your masked MoCoV3 pipeline.
    Accepts (x, mask), but ignores mask internally.
    Always returns (cls_token, pooled_token) tuple to match transformer interface.
    """

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

        # Handle cases where fc exists (for both timm or torchvision models)
        if hasattr(resnet, "fc"):
            self.output_dim = resnet.fc.in_features
            self.resnet.fc = nn.Identity()
        elif hasattr(resnet, "classifier"):
            self.output_dim = resnet.classifier.in_features
            self.resnet.classifier = nn.Identity()
        else:
            raise ValueError("ResNet model does not have fc or classifier attribute")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        features = self.resnet(x)
        return features, None  # cleaner interface for downstream logic