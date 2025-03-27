from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import torch


class ModelAndBackboneCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        # Ensure we don't auto-insert metric names, so your original placeholders remain
        super().__init__(auto_insert_metric_name=False, *args, **kwargs)

    def _save_checkpoint(self, trainer, filepath):
        # First save the full model checkpoint
        super()._save_checkpoint(trainer, filepath)

        # Then save only the backbone under a separate file
        backbone_path = Path(filepath).with_stem(
            f"{Path(filepath).stem}_backbone")
        torch.save(trainer.model.backbone.state_dict(), backbone_path)
