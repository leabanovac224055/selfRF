from typing import Dict, Callable, Union

from torchsig.transforms.dataset_transforms import ComplexTo2D, Transform
from torchsig.transforms.base_transforms import Compose
from torchsig.transforms.target_transforms import ClassIndex, FamilyIndex

from selfrf.transforms import (
    ToSpectrogramTensor,
    Identity,
    ToTensor,
    SpectrogramImageHighQuality,
    BYOLTransform,
    DINOTransform,
)
from selfrf.pretraining.config import BaseConfig, TrainingConfig, EvaluationConfig
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import TransformType, SSLModelType, DatasetType


class TransformFactory:
    @staticmethod
    def create_spectrogram_transform(config: BaseConfig) -> Transform:
        return Compose([
            SpectrogramImageHighQuality(
                nfft=config.nfft,
            ),
        ])

    @staticmethod
    def create_iq_transform(config: BaseConfig) -> Transform:
        return Compose([
            ComplexTo2D(),
            ToTensor(to_float_32=config.to_float_32),
        ])

    _transform_registry: Dict[TransformType, Callable[[BaseConfig], Transform]] = {
        TransformType.SPECTROGRAM: create_spectrogram_transform,
        TransformType.IQ: create_iq_transform
    }

    _ssl_transform_registry: Dict[SSLModelType, Callable] = {
        SSLModelType.BYOL: BYOLTransform,
        SSLModelType.DINO: DINOTransform,
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

        # wrap tensor transform with SSL transform
        return cls._ssl_transform_registry[config.ssl_model](
            tensor_transform=tensor_transform
        )

    @classmethod
    def create_target_transform(cls, config: BaseConfig) -> Transform:
        if config.dataset == DatasetType.TORCHSIG_NARROWBAND:
            if config.family:
                return Compose([
                    FamilyIndex(class_list=get_class_list(config)),
                    ClassIndex(),
                ])
            return ClassIndex()

        if config.dataset == DatasetType.TORCHSIG_WIDEBAND:
            return [Identity()]


def build_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    return TransformFactory.create_transform(config)


def build_target_transform(config: BaseConfig) -> Transform:
    return TransformFactory.create_target_transform(config)
