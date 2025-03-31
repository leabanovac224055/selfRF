from dataclasses import dataclass
from typing import Dict, Type, Callable, Union
import torch

from selfrf.pretraining.config import TrainingConfig, BaseConfig
from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.models.iq_models import build_resnet1d
from selfrf.models.spectrogram_models import build_resnet2d, build_vit
from selfrf.models.ssl_models import BYOL, DINO, DenseCL
from selfrf.pretraining.utils.enums import BackboneArchitecture, SSLModelType
from selfrf.models.iq_models.xcit.xcit1d import XCiT1d


@dataclass(frozen=True)  # makes the dataclass immutable
class BackboneConfig:
    backbone_arch: BackboneArchitecture
    is_spectrogram: bool


class ModelFactory:
    _backbone_registry: Dict[BackboneConfig, Callable] = {
        BackboneConfig(BackboneArchitecture.RESNET, False): lambda **kwargs: build_resnet1d(input_channels=2, ** kwargs),
        BackboneConfig(BackboneArchitecture.RESNET, True): lambda **kwargs: build_resnet2d(input_channels=1, ** kwargs),
        BackboneConfig(BackboneArchitecture.VIT, True): lambda **kwargs: build_vit(input_channels=1, ** kwargs),
        BackboneConfig(BackboneArchitecture.XCIT, False): lambda **kwargs: XCiT1d(input_channels=2, ** kwargs),
    }

    _ssl_registry: Dict[SSLModelType, Type] = {
        SSLModelType.BYOL: BYOL,
        SSLModelType.DINO: DINO,
        SSLModelType.DENSECL: DenseCL,
    }

    @classmethod
    def create_backbone(cls, config: Union[TrainingConfig, EvaluationConfig]) -> torch.nn.Module:
        """Create backbone from config"""
        backbone_arch = config.backbone.get_architecture()
        is_spectrogram = config.spectrogram
        backbone_config = BackboneConfig(
            backbone_arch=backbone_arch,
            is_spectrogram=is_spectrogram
        )

        try:
            builder = cls._backbone_registry[backbone_config]
        except KeyError:
            # Create a more informative error message
            data_type = "spectrogram" if is_spectrogram else "IQ"

            # Get available configurations
            available_configs = []
            for cfg in cls._backbone_registry.keys():
                data_str = "spectrogram" if cfg.is_spectrogram else "IQ"
                available_configs.append(
                    f"{cfg.backbone_arch.value} with {data_str} data")

            raise ValueError(
                f"No backbone implementation found for '{backbone_arch.name}' architecture with {data_type} data.\n"
                f"Make sure 'spectrogram={is_spectrogram}' is compatible with your backbone choice.\n"
                f"Available combinations:\n" +
                "\n".join(f"- {c}" for c in available_configs)
            )

        if isinstance(config, EvaluationConfig):
            return builder(
                version=config.backbone.get_size().value,
                provider=config.backbone_provider,
            )

        return builder(
            version=config.backbone.get_size().value,
            provider=config.backbone_provider,
            features_only=True if config.ssl_model == SSLModelType.DENSECL else False,
        )

    @classmethod
    def create_ssl_model(cls, config: TrainingConfig) -> torch.nn.Module:
        """Create SSL model from config"""
        backbone = cls.create_backbone(config)
        ssl_type = SSLModelType(config.ssl_model)
        ssl_model = cls._ssl_registry[ssl_type]
        return ssl_model(
            backbone=backbone,
            batch_size_per_device=config.batch_size,
            use_online_linear_eval=config.online_linear_eval,
            num_classes=len(get_class_list(config))
        )


def build_backbone(config: BaseConfig) -> torch.nn.Module:
    return ModelFactory.create_backbone(config)


def build_ssl_model(config: TrainingConfig) -> torch.nn.Module:
    return ModelFactory.create_ssl_model(config)
