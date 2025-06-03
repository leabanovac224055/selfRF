import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
        "/home/airbus/selfRF/train/moco_v3/lightning_logs/version_2/MOCOV3-xcit-xcit_tiny_12_p16_224-IQDM_NARROWBAND-iq-e45-b64-loss3.819.ckpt",
        map_location=config.device,
        weights_only=False
    )
    ssl_model.load_state_dict(iq_ckpt["state_dict"], strict=False)

    # 🔍 Determine IQ encoder embedding dim
    dummy_input = torch.randn(1, 2, config.num_iq_samples).to(config.device)
    with torch.no_grad():
        dummy_output = iq_encoder(dummy_input)
    iq_embedding_dim = dummy_output.shape[1]
    print(f"🔢 IQ encoder output dim: {iq_embedding_dim}")

    # ✅ Build and load frozen metadata tower
    metadata_model = MLP(
        input_dim=config.metadata_input_dim,
        hidden_dim=config.metadata_hidden_dim,
        output_dim=config.metadata_output_dim,
    )
    metadata_model.to(config.device)
    metadata_ckpt = torch.load(
        "/home/airbus/selfRF/train/metadata_tower/lightning_logs/version_3/metadata_tower-TWO_TOWER_NARROWBAND-iq-eepoch=14-b16-losstrain_loss=2.714.ckpt",
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

    # ✅ Setup trainer
    logger = TensorBoardLogger("tb_logs", name="fusion_train")
    trainer = Trainer(
        max_epochs=config.two_tower_num_epochs,
        logger=logger,
        accelerator=config.device.type,
        precision="16-mixed",
        devices=1,
    )

    # 🚀 Train fusion head
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_fusion()
    
# Note: Each IQ view undergoes augmentations such as frequency shifting and spectral inversion.
# The metadata vector (original center frequency, bandwidth, duration) is *not* updated to reflect these changes.
# This design encourages the model to learn embeddings that are invariant to view-specific distortions,
# while grounding each view in the same real-world signal identity.