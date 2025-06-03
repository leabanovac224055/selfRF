import os
from dotenv import load_dotenv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm

from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.factories import (
    build_dataloader,
    build_ssl_model,
    build_meta_model,
    build_fusion_head,
)
from selfrf.data import TwoTowerTrainModule
from selfrf.pretraining.utils.callbacks import ModelAndBackboneCheckpoint
from selfrf.models.ssl_models import BYOL, DINO
from selfrf.pretraining.utils.enums import SSLModelType, BackboneType


def train(config: TrainingConfig):

    # Phase 1: SSL backbone training
    datamodule = build_dataloader(config)
    datamodule.prepare_data()
    datamodule.setup()

    if not config.online_linear_eval:
        datamodule.val_dataloader = None

    ssl_model = build_ssl_model(config)

    # Logger
    logger = TensorBoardLogger(
        os.path.join(config.training_path, config.ssl_model.value)
    )

    # Checkpoint saver
    checkpoint_callback = ModelAndBackboneCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            f"wbesttrainloss-"
            f"{config.ssl_model.name}"
            f"-{config.backbone.value}"
            f"-{config.dataset.value}"
            f"-{'spec' if config.spectrogram else 'iq'}"
            f"-e{{epoch:d}}"
            f"-b{config.batch_size}"
            f"-loss{{train_loss:.3f}}"
        ),
        save_top_k=10,
        verbose=True,
        save_last=True,
        monitor="train_loss",
        mode="min",
    )
    
    early_stopping = EarlyStopping(
            monitor='train_loss',     # 🟢 Use val_loss instead of train_loss
            patience=20,            # Adjust patience as needed
            verbose=True,
            mode="min"
        )

    trainer = Trainer(
        precision='16-mixed',
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
    )

    trainer.fit(model=ssl_model, datamodule=datamodule)
    

if __name__ == '__main__':
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    train(config)