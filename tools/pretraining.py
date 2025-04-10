import os
from dotenv import load_dotenv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.factories import (
    build_dataloader,
    build_ssl_model,
    build_meta_model,
    build_fusion_head,
)
from selfrf.data import TwoTowerTrainModule
from selfrf.pretraining.utils.callbacks import ModelAndBackboneCheckpoint


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

    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model=ssl_model, datamodule=datamodule)

    # Phase 2: Train metadata tower + fusion
    if config.train_two_tower_after_ssl:
        print("🧠 Starting two-tower training phase (metadata + fusion)...")

        # ✅ Reuse best (or last) checkpoint from SSL phase
        ssl_ckpt_path = checkpoint_callback.last_model_path
        if not ssl_ckpt_path:
            raise RuntimeError("No SSL checkpoint found to resume from.")

        print(f"✅ Using frozen SSL encoder from checkpoint: {ssl_ckpt_path}")
        ssl_model = build_ssl_model(config).load_from_checkpoint(ssl_ckpt_path)

        # Freeze IQ encoder
        iq_encoder = ssl_model.backbone.eval()
        for param in iq_encoder.parameters():
            param.requires_grad = False

        # Dynamically infer IQ embedding dim
        with torch.no_grad():
            dummy_input = torch.randn(
                1, 2, config.num_iq_samples).to(config.device)
            dummy_output = iq_encoder(dummy_input)
            iq_dim = dummy_output.shape[1]

        # Metadata tower + fusion head
        metadata_tower = build_meta_model(config)
        fusion_head = build_fusion_head(
            fusion_type="concat_mlp",
            iq_dim=iq_dim,
            meta_dim=config.metadata_output_dim,
        )
        # Loss: contrastive (same view 1 + meta vs. view 2 + meta)
        loss_fn = torch.nn.MSELoss()  # or any other contrastive loss you prefer

        model = TwoTowerTrainModule(
            iq_encoder=iq_encoder,
            metadata_tower=metadata_tower,
            fusion_head=fusion_head,
            loss_fn=loss_fn,
            lr=1e-3,
        )

        # Reuse dataloader (can be different dataset if needed)
        datamodule = build_dataloader(config)
        datamodule.prepare_data()
        datamodule.setup()

        # Checkpoint for two-tower phase
        two_tower_ckpt = ModelCheckpoint(
            dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}/two_tower",
            filename="two_tower-e{epoch}-loss{train_loss:.3f}",
            save_top_k=5,
            verbose=True,
            save_last=True,
            monitor="train_loss",
            mode="min",
        )

        trainer = Trainer(
            max_epochs=config.num_epochs,
            devices=1,
            accelerator=config.device.type,
            logger=logger,
        )

        trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    train(config)
