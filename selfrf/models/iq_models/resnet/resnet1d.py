import timm
from torch.nn import Linear

from torchsig.models.model_utils.model_utils_1d.conversions_to_1d import convert_2d_model_to_1d

from selfrf.pretraining.utils.enums import BackboneProvider

__all__ = ["build_resnet1d"]


def build_resnet1d(
    input_channels: int,
    n_features: int = 2048,
    version: str = "18",
    provider: BackboneProvider = BackboneProvider.TIMM,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    feature_only=False,
):
    """Constructs and returns a 1d version of the ResNet model.
    Args:

        input_channels (int):
            Number of 1d input channels; e.g., common practice is to split complex number time-series data into 2 channels, representing the real and imaginary parts respectively

        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        version (str):
            Specifies the version of resnet to use. See the timm resnet documentation for details. Examples are '18', '34' or'50'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """

    if provider is BackboneProvider.TIMM:
        mdl = convert_2d_model_to_1d(
            timm.create_model(
                "resnet" + version,
                in_chans=input_channels,
                drop_path_rate=drop_path_rate,
                drop_rate=drop_rate,
                features_only=feature_only
            )
        )
        if not feature_only:
            mdl.fc = Linear(mdl.fc.in_features, n_features)   
        return mdl
    else:
        raise ValueError(f"{provider} does not provider a ResNet 1D backbone.")
