from dataclasses import dataclass
from typing import Dict, Callable, Union
import numpy as np

from torchsig.transforms.dataset_transforms import ComplexTo2D, Spectrogram, Transform
from torchsig.transforms.base_transforms import Normalize, Compose
from torchsig.transforms.target_transforms import ClassIndex, FamilyIndex


from selfrf.transforms import (
    ToSpectrogramTensor,
    ToTensor,
    BYOLTransform,
    SpectrogramImage,
)
from selfrf.pretraining.config import BaseConfig, TrainingConfig, EvaluationConfig
from selfrf.transforms.extra.target_transforms import ConstantTargetTransform
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import TransformType, SSLModelType, DatasetType


@dataclass
class SpectrogramConfig:
    nfft: int
    noverlap: int
    mode: str = 'psd'


class TransformFactory:
    @staticmethod
    def create_spectrogram_transform(config: BaseConfig) -> Transform:
        return Compose([
            Spectrogram(
                fft_size=config.nfft,
            ),
            Normalize(norm=np.inf, flatten=True),
            SpectrogramImage(normalize_max=1),
            ToSpectrogramTensor(
                to_float_32=config.to_float_32,
            ),
        ])

    @staticmethod
    def create_iq_transform(config: BaseConfig) -> Transform:
        return Compose([
            Normalize(norm=np.inf),
            ComplexTo2D(),
            ToTensor(to_float_32=config.to_float_32),
        ])

    _transform_registry: Dict[TransformType, Callable[[BaseConfig], Transform]] = {
        TransformType.SPECTROGRAM: create_spectrogram_transform,
        TransformType.IQ: create_iq_transform
    }

    _ssl_transform_registry: Dict[SSLModelType, Callable] = {
        SSLModelType.BYOL: BYOLTransform
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
                    FamilyIndex(class_list=get_class_list(config))
                ])
            return ClassIndex(class_list=get_class_list(config))

        if config.dataset == DatasetType.TORCHSIG_WIDEBAND:
            # the wideband dataset cannot be used for online linear evaluation
            # therefore, we map the targets to a constant value
            # in order to unequal batch sizes
            return ConstantTargetTransform(0)


def build_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    return TransformFactory.create_transform(config)


def build_target_transform(config: BaseConfig) -> Transform:
    return TransformFactory.create_target_transform(config)
