from typing import Any

from torchsig.signals.signal_lists import TorchSigSignalLists
from torch.utils.data import Subset
from selfrf.pretraining.utils.enums import DatasetType


def get_class_list(config) -> list:
    from selfrf.pretraining.factories import build_dataloader
    # Return cached version if available
    if hasattr(config, "_cached_class_list"):
        return config._cached_class_list

    # ✅ Use TorchSig labels
    if config.dataset in {DatasetType.TORCHSIG_NARROWBAND, DatasetType.TORCHSIG_WIDEBAND}:
        class_list = (
            TorchSigSignalLists.family_list if config.family
            else TorchSigSignalLists.all_signals
        )
        config._cached_class_list = class_list
        return class_list

    # ✅ Use Zarr-based real class names
    datamodule = build_dataloader(config)
    datamodule.setup("fit")
    dataset = datamodule.train_dataloader().dataset

    # Unwrap if it's a Subset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    # Extract class_name from Zarr metadata
    if hasattr(dataset, "metadata"):
        class_map = {}
        for v in dataset.metadata.values():
            ann = v[0]
            idx = ann.get("class_index")
            name = ann.get("class_name")
            if idx is not None and name is not None:
                class_map[idx] = name

        class_list = [class_map[i] for i in sorted(class_map)]
        print(f"✅ Loaded {len(class_list)} real class names from Zarr")
        config._cached_class_list = class_list
        return class_list

    # 🚫 No fallback — raise error instead of using dummy labels
    raise RuntimeError(
        "❌ Could not determine class list: dataset has no TorchSig labels and no Zarr metadata with class names"
    )



class Signal:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata
