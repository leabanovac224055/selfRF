from enum import Enum


from dataclasses import dataclass


class BackboneArchitecture(Enum):
    """Base backbone architecture types."""
    RESNET = "resnet"
    VIT = "vit"

    def __str__(self) -> str:
        return self.name


class BackboneSize(Enum):
    """Size variants for different backbone architectures."""
    # ResNet variants
    RESNET_18 = "18"
    RESNET_34 = "34"
    RESNET_50 = "50"
    RESNET_101 = "101"
    RESNET_152 = "152"

    # ViT variants
    VIT_BASE = "b"
    VIT_LARGE = "l"
    VIT_HUGE = "h"

    def __str__(self) -> str:
        return self.name


class BackboneProvider(Enum):
    """Library/framework providing the backbone implementation."""
    TIMM = "timm"
    DETECTRON2 = "detectron2"
    TORCHSIG = "torchsig"

    @classmethod
    def from_string(cls, name: str) -> 'BackboneProvider':
        """Convert string to BackboneProvider enum."""
        try:
            # Try direct match with enum name
            return cls[name.upper()]
        except KeyError:
            # Try with the value
            try:
                return cls(name.lower())
            except ValueError:
                # If all else fails
                valid_providers = [f"{p.name} ({p.value})" for p in cls]
                raise ValueError(
                    f"Unknown backbone provider: '{name}'. Valid values are: {', '.join(valid_providers)}"
                )

    def __str__(self) -> str:
        return self.name


@dataclass
class BackboneSpec:
    """Specification of a backbone model architecture and size.

    Combines architecture and size into a single specification.
    The provider is configured separately for more flexibility.
    """
    architecture: BackboneArchitecture
    size: BackboneSize

    def __str__(self) -> str:
        """Get a human-readable string representation."""
        return f"{self.architecture.value}-{self.size.value}"

    @classmethod
    def from_string(cls, spec_string: str) -> 'BackboneSpec':
        """Parse a backbone specification from a string.

        Format: "architecture-size"
        Examples: "resnet-50", "vit-b"
        """
        parts = spec_string.lower().split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid backbone spec format: {spec_string}. "
                             f"Expected format: 'architecture-size'")

        arch_str, size_str = parts[0], parts[1]

        # Find matching architecture
        try:
            architecture = BackboneArchitecture(arch_str)
        except ValueError:
            raise ValueError(f"Unknown architecture: {arch_str}")

        # Find matching size (need to check prefix)
        size = None
        for size_enum in BackboneSize:
            if size_enum.name.lower().startswith(f"{arch_str}_{size_str}"):
                size = size_enum
                break

        if size is None:
            raise ValueError(
                f"Unknown size '{size_str}' for architecture '{arch_str}'")

        return cls(architecture, size)


# For backward compatibility with existing code
class BackboneType(Enum):
    """Backbone types for model training."""
    RESNET18 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_18)
    RESNET34 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_34)
    RESNET50 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_50)
    RESNET101 = BackboneSpec(
        BackboneArchitecture.RESNET, BackboneSize.RESNET_101)
    VIT_B = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_BASE)
    VIT_L = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_LARGE)
    VIT_H = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_HUGE)

    def get_architecture(self) -> BackboneArchitecture:
        """Get the backbone specification."""
        return self.value.architecture

    def get_size(self) -> BackboneSize:
        """Get the backbone size."""
        return self.value.size

    @classmethod
    def from_string(cls, name: str) -> 'BackboneType':
        """Convert string to BackboneType enum."""
        try:
            # Try direct match with enum name
            return cls[name.upper()]
        except KeyError:
            # Try case-insensitive match
            for bt in cls:
                if bt.name.lower() == name.lower():
                    return bt
            raise ValueError(
                f"Unknown backbone type: {name}. Valid values are: {[t.name for t in cls]}")

    def __str__(self) -> str:
        return self.name


class SSLModelType(Enum):
    BYOL = "byol"


class DatasetType(Enum):
    TORCHSIG_NARROWBAND = "narrowband"
    TORCHSIG_WIDEBAND = "wideband"


class TransformType(Enum):
    SPECTROGRAM = "spectrogram"
    IQ = "iq"
