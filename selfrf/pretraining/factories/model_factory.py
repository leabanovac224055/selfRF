from dataclasses import dataclass
from typing import Dict, Type, Callable, Union, Optional
import torch

from selfrf.pretraining.config import TrainingConfig, BaseConfig
from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.models.iq_models import build_resnet1d, ResNetWrapper, XCiT1d
from selfrf.models.spectrogram_models import build_resnet2d, build_vit
from selfrf.models.ssl_models import BYOL, DINO, DenseCL, MoCoV3
from selfrf.pretraining.utils.enums import BackboneArchitecture, SSLModelType
from selfrf.models.meta_models import MLP, ConcatMLPHead
from selfrf.pretraining.config.training_config import TrainingStage


@dataclass(frozen=True)  # makes the dataclass immutable
class BackboneConfig:
    backbone_arch: BackboneArchitecture
    is_spectrogram: bool


class ModelFactory:
    _backbone_registry: Dict[BackboneConfig, Callable] = {
        BackboneConfig(BackboneArchitecture.RESNET, False): lambda **kwargs: ResNetWrapper(build_resnet1d(input_channels=2, ** kwargs)),
        BackboneConfig(BackboneArchitecture.RESNET, True): lambda **kwargs: build_resnet2d(input_channels=1, ** kwargs),
        BackboneConfig(BackboneArchitecture.VIT, True): lambda **kwargs: build_vit(input_channels=1, ** kwargs),
        BackboneConfig(BackboneArchitecture.XCIT, False): lambda **kwargs: XCiT1d(input_channels=2, ** kwargs),
    }

    _ssl_registry: Dict[SSLModelType, Type] = {
        SSLModelType.BYOL: BYOL,
        SSLModelType.DINO: DINO,
        SSLModelType.DENSECL: DenseCL,
        SSLModelType.MOCOV3: MoCoV3
    }

    # Metadata model registry
    _meta_registry: Dict[str, Callable] = {
        "mlp": lambda config: MLP(
            input_dim=config.metadata_input_dim,
            hidden_dim=config.metadata_hidden_dim,
            output_dim=config.metadata_output_dim,
        )
    }

    # Fusion head registry
    _fusion_head_registry: Dict[str, Callable] = {
        "concat_mlp": lambda iq_dim, meta_dim, hidden_dim=256, output_dim=128: ConcatMLPHead(
            iq_embedding_dim=iq_dim,
            meta_embedding_dim=meta_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
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
            feature_only=True if config.ssl_model == SSLModelType.DENSECL else False,
        )

    @classmethod
    def create_ssl_model(cls, config: TrainingConfig) -> torch.nn.Module:
        """Create SSL model from config"""
        backbone = cls.create_backbone(config)
        ssl_type = SSLModelType(config.ssl_model)
        
        ssl_model = cls._ssl_registry[ssl_type]
        
        # Handle MoCo model
        if ssl_type == SSLModelType.MOCOV3:
            return ssl_model(
                num_classes=len(get_class_list(config)),
                batch_size_per_device=config.batch_size,
                backbone=backbone,
                dim=config.projection_dim,
                mlp_dim=config.mlp_dim,
                T=config.temperature,
                use_online_linear_eval=config.online_linear_eval,
                use_masked_pooling=True
            )

        return ssl_model(
            backbone=backbone,
            batch_size_per_device=config.batch_size,
            use_online_linear_eval=config.online_linear_eval,
            num_classes=len(get_class_list(config))
        )

    @classmethod
    def create_meta_model(cls, config: BaseConfig) -> Optional[torch.nn.Module]:
        if getattr(config, "training_stage", None) != TrainingStage.METADATA and getattr(config, "training_stage", None) != TrainingStage.FUSION:
            return None
        return cls._meta_registry[config.metadata_model.value.lower()](config)

    @classmethod
    def create_fusion_head(
        cls,
        fusion_type: str,
        iq_dim: int,
        meta_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128
    ) -> torch.nn.Module:
        return cls._fusion_head_registry[fusion_type](iq_dim, meta_dim, hidden_dim, output_dim)


def build_backbone(config: BaseConfig) -> torch.nn.Module:
    return ModelFactory.create_backbone(config)


def build_ssl_model(config: TrainingConfig) -> torch.nn.Module:
    return ModelFactory.create_ssl_model(config)


def build_meta_model(config: BaseConfig) -> Optional[torch.nn.Module]:
    return ModelFactory.create_meta_model(config)


def build_fusion_head(
    fusion_type: str,
    iq_dim: int,
    meta_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 128
) -> torch.nn.Module:
    return ModelFactory.create_fusion_head(fusion_type, iq_dim, meta_dim, hidden_dim, output_dim)
