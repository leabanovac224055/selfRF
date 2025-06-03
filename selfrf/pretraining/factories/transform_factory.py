from typing import Dict, Callable, Union, Optional, List 
import numpy as np
from torchsig.transforms.dataset_transforms import ComplexTo2D, Transform
from torchsig.transforms.base_transforms import Compose, Normalize
from torchsig.transforms.target_transforms import ClassIndex, FamilyIndex, ClassName
from torchsig.signals.signal_lists import TorchSigSignalLists


from selfrf.transforms import (
    Identity,
    ToTensor,
    SpectrogramImageHighQuality,
    BYOLTransform,
    DINOTransform,
    DenseCLTransform,
    MoCoTransform
)
from selfrf.pretraining.config import BaseConfig, TrainingConfig, EvaluationConfig
from selfrf.pretraining.utils.enums import TransformType, SSLModelType, DatasetType
from selfrf.transforms.extra.transforms import RandomAWGN


class TransformFactory:
    @staticmethod
    def create_spectrogram_transform(config: BaseConfig) -> Transform:
        return Compose([
            RandomAWGN(noise_power_db=(-100, -80)),
            Normalize(norm=np.inf),
            SpectrogramImageHighQuality(
                nfft=config.nfft,
            ),
        ])

    @staticmethod
    def create_iq_transform(config: BaseConfig) -> Transform:
        return Compose([
            RandomAWGN(noise_power_db=(-80, -80)),
            Normalize(norm=np.inf),
            ComplexTo2D(),
            ToTensor(),
        ])

    _transform_registry: Dict[TransformType, Callable[[BaseConfig], Transform]] = {
        TransformType.SPECTROGRAM: create_spectrogram_transform,
        TransformType.IQ: create_iq_transform
    }

    _ssl_transform_registry: Dict[str, Callable] = {
        "byol": BYOLTransform,
        "dino": DINOTransform,
        "densecl": DenseCLTransform,
        "moco_v3": MoCoTransform  # Placeholder for MoCoV3
    }

    @classmethod
    def create_tensor_transform(cls, config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
        transform_type = TransformType.SPECTROGRAM if config.spectrogram else TransformType.IQ
        return cls._transform_registry[transform_type](config)

    @classmethod
    def create_transform(cls, config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
        tensor_transform = cls.create_tensor_transform(config)

        if isinstance(config, EvaluationConfig):
            return tensor_transform

        return cls._ssl_transform_registry[config.ssl_model.value](
            tensor_transform=tensor_transform
        )


    @classmethod
    def create_target_transform(cls, config: BaseConfig) -> List[Transform]:
        # Narrowband: ClassIndex → ClassName, or FamilyIndex
        if config.dataset in {
            DatasetType.TORCHSIG_NARROWBAND,
            DatasetType.IQDM_NARROWBAND
        }:
            if config.family:
                return [
                    FamilyIndex(
                        class_family_dict=TorchSigSignalLists.family_dict,
                        family_list=TorchSigSignalLists.family_list
                    )
                ]
            else:
                return [ClassIndex(), ClassName()]

        # Wideband: no-op
        if config.dataset in {
            DatasetType.TORCHSIG_WIDEBAND,
            DatasetType.IQDM_WIDEBAND
        }:
            return [Identity()]


def build_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    return TransformFactory.create_transform(config)


def build_target_transform(config: BaseConfig) -> List[Transform]:
    return TransformFactory.create_target_transform(config)