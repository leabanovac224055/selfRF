from typing import Any

from torchsig.signals.signal_lists import TorchSigSignalLists


def get_class_list(config: Any) -> list:
    if config.family:
        return TorchSigSignalLists.family_list
    return TorchSigSignalLists.all_signals
