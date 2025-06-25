import torch
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from selfrf.pretraining.utils.callbacks import ModelCheckpoint

from selfrf.data import TwoTowerTrainModule
from selfrf.models.meta_models.mlp import MLP
from selfrf.models.meta_models import ConcatMLPHead
from selfrf.pretraining.factories import build_ssl_model, build_dataloader
from selfrf.pretraining.config import parse_training_config, print_config
from selfrf.pretraining.config.training_config import TrainingStage


def train_fusion():
    config = parse_training_config()
    print_config(config)

    # ✅ Set training stage for fusion
    config.training_stage = TrainingStage.FUSION

    # ✅ Build frozen IQ encoder (pretrained SSL model)
    ssl_model = build_ssl_model(config)
    iq_encoder = ssl_model.backbone
    iq_encoder.to(config.device)

    # ✅ Load IQ encoder checkpoint
    iq_ckpt = torch.load(
        "/home/airbus/selfRF/train/moco_v3/lightning_logs/version_21/combined-dataset-MOCOV3-resnet-50-IQDM_NARROWBAND-iq-e38-b64-loss0.706.ckpt",
        map_location=config.device,
        weights_only=False
    )
    ssl_model.load_state_dict(iq_ckpt["state_dict"], strict=False)

    # 🔍 Determine IQ encoder embedding dim
    dummy_input = torch.randn(1, 2, config.num_iq_samples).to(config.device)
    with torch.no_grad():
        cls_token, pooled_token = iq_encoder(dummy_input)
        if pooled_token is not None:
            features = torch.cat([cls_token, pooled_token], dim=-1)
        else:
            features = cls_token

    iq_embedding_dim = features.shape[1]
    print(f"🔢 IQ encoder output dim: {iq_embedding_dim}")

    # ✅ Build and load frozen metadata tower
    metadata_model = MLP(
        input_dim=config.metadata_input_dim,
        hidden_dim=config.metadata_hidden_dim,
        output_dim=config.metadata_output_dim,
    )
    metadata_model.to(config.device)
    metadata_ckpt = torch.load(
        "/home/airbus/selfRF/train/metadata_tower/lightning_logs/version_15/metadata_tower-TWO_TOWER_NARROWBAND-iq-sepoch=40-b64-losstrain_loss=4.110.ckpt",
        map_location=config.device
    )
    metadata_model.load_state_dict(metadata_ckpt["state_dict"], strict=False)

    # ✅ Build fusion head with correct input dim
    fusion_head = ConcatMLPHead(
        iq_embedding_dim=iq_embedding_dim,
        meta_embedding_dim=config.metadata_output_dim,
    )
    print(f"✅ Fusion head input dim: {iq_embedding_dim + config.metadata_output_dim}")

    # ✅ Wrap in LightningModule
    model = TwoTowerTrainModule(
        iq_encoder=iq_encoder,
        metadata_tower=metadata_model,
        fusion_head=fusion_head,
        lr=1e-3,
        temperature=0.1,
        freeze_iq_encoder=False,
        freeze_metadata_tower=False
    )

    # ✅ Build data loaders
    datamodule = build_dataloader(config)
    datamodule.setup("fit")

    # Logger
    logger = TensorBoardLogger(
        os.path.join(config.training_path, "fusion_head")
    )

    # ✅ Checkpoint callback — ONLY saves full model
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            f"fusion_head-"
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

    # ✅ Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=10,
        verbose=True,
    )

    # ✅ Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        precision=32,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
    )

    # 🚀 Train fusion head
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_fusion()
    
# Note: Each IQ view undergoes augmentations such as frequency shifting and spectral inversion.
# The metadata vector (original center frequency, bandwidth, duration) is *not* updated to reflect these changes.
# This design encourages the model to learn embeddings that are invariant to view-specific distortions,
# while grounding each view in the same real-world signal identity.