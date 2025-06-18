import os
import torch
#from selfrf.data.meta.meta_tower import MetadataTrainModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from selfrf.pretraining.factories import build_dataloader
from selfrf.models.meta_models import MLP
from selfrf.data import MetadataTrainModule
from dotenv import load_dotenv
from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.config.training_config import TrainingStage

import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer

def train_metadata_tower(config: TrainingConfig):
    print("🚀 Starting Metadata Tower Training...")
    print(f"✅ Using device: {config.device}")

    # Load the metadata tower model
    metadata_tower = MLP(
        input_dim=config.metadata_input_dim,
        hidden_dim=config.metadata_hidden_dim,
        output_dim=config.metadata_output_dim
    )
    print("✅ Metadata tower model initialized")

    # Create the metadata training module
    model = MetadataTrainModule(metadata_tower)

    # Prepare the data module
    datamodule = build_dataloader(config)
    datamodule.prepare_data()
    datamodule.setup()
    print("✅ DEBUG: Data module prepared")

    # Setup logging and checkpointing
    logger = TensorBoardLogger(os.path.join(config.training_path, "metadata_tower"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            "metadata_tower"
            f"-{config.dataset.value}"
            f"-{'spec' if config.spectrogram else 'iq'}"
            f"-s{{epoch:d}}"
            f"-b{config.batch_size}"
            f"-loss{{train_loss:.3f}}"
        ),
        save_top_k=10,         # Keeps the top 5 checkpoints
        save_last=True,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )


    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    # Trainer configuration
    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )

    print("🚀 Training the metadata tower...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    # Override the mode specifically for metadata training
    config.training_stage = TrainingStage.METADATA
    is_val = False  # Not in validation mode for training
    print(f"✅ Mode set to: {config.training_stage}")
    train_metadata_tower(config)