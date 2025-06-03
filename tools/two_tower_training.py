import os
from dotenv import load_dotenv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn as nn

from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_meta_model, build_fusion_head, build_backbone
from selfrf.data import TwoTowerTrainModule
from selfrf.models.ssl_models import BYOL, DINO
from selfrf.pretraining.utils.enums import SSLModelType

def train_two_tower(config: TrainingConfig):
    print("🚀 DEBUG: Starting two-tower training...")
    print(f"✅ DEBUG: Loading SSL encoder from checkpoint: {config.ssl_model_path}")
    
    # Load the checkpoint manually with appropriate safe globals and weights_only=False
    try:
        print("🔄 DEBUG: Attempting to load checkpoint with weights_only=False...")
        # Allow the specific numpy scalar function required for loading
        torch.serialization.add_safe_globals([("numpy.core.multiarray", "scalar")])
        checkpoint = torch.load(config.ssl_model_path, map_location='cuda', weights_only=False)
        print("✅ DEBUG: Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load checkpoint: {e}")
        return

    # Determine model class based on configuration
    model_class = DINO if config.ssl_model == SSLModelType.DINO else BYOL
    print(f"🔄 DEBUG: Selected model class: {model_class.__name__}")

    # Extract hyperparameters from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})
    print(f"🔍 DEBUG: Extracted hyperparameters: {hparams}")

    # Rebuild the SSL model
    try:
        ssl_model = model_class(
            num_classes=hparams.get('num_classes', 57),
            batch_size_per_device=hparams.get('batch_size_per_device', 16),
            backbone=config.backbone,
            num_ftrs=hparams.get('num_ftrs', 2048),
            hidden_dim=hparams.get('hidden_dim', 4096),
            out_dim=hparams.get('out_dim', 128),
            momentum_teacher=hparams.get('momentum_teacher', 0.996),
            use_online_linear_eval=hparams.get('use_online_linear_eval', False)
        )
        ssl_model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ DEBUG: SSL model initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load SSL model from checkpoint: {e}")
        return

    # Extract the backbone type from the SSL model or checkpoint
    try:
        print(f"🔄 DEBUG: Building backbone using config: {config.backbone}")
        iq_encoder = build_backbone(config)  # Pass the whole config object
        iq_encoder.eval()
        for param in iq_encoder.parameters():
            param.requires_grad = False
        print("✅ DEBUG: Backbone set to eval mode and frozen")
    except Exception as e:
        print(f"❌ ERROR: Failed to build and freeze backbone: {e}")

    # Dynamically determine iq_dim
    try:
        with torch.no_grad():
            # Move the backbone to the same device as the input tensor
            iq_encoder = iq_encoder.to(config.device)

            dummy_input = torch.randn(1, 2, config.num_iq_samples).to(config.device)
            iq_output = iq_encoder(dummy_input)
            iq_dim = iq_output.shape[1]  # Extract dimension
        print(f"✅ DEBUG: IQ Embedding Dimension inferred as: {iq_dim}")
    except Exception as e:
        print(f"❌ ERROR: Failed to infer IQ embedding dimension: {e}")
        return

    # Build the metadata tower and fusion head
    metadata_tower = build_meta_model(config)
    fusion_head = build_fusion_head("concat_mlp", iq_dim=iq_dim, meta_dim=config.metadata_output_dim)
    print("✅ DEBUG: Metadata tower and fusion head initialized")

    # Define the two-tower model
    model = TwoTowerTrainModule(
        iq_encoder=iq_encoder,
        metadata_tower=metadata_tower,
        fusion_head=fusion_head,
        loss_fn=torch.nn.MSELoss(),
        lr=1e-3,
    )

    # Prepare the data module
    datamodule = build_dataloader(config)
    datamodule.prepare_data()
    datamodule.setup()
    print("✅ DEBUG: Data module prepared")

    # Setup logging and checkpointing
    logger = TensorBoardLogger(os.path.join(config.training_path, "two_tower"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}/two_tower",
        filename="two_tower-e{epoch}-loss{train_loss:.3f}",
        save_top_k=5,
        verbose=True,
        save_last=True,
        monitor="train_loss",
        mode="min",
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="train_loss",   # Monitor the training loss
        mode="min",             # Stop when loss stops decreasing
        patience=10,            # Number of epochs with no improvement after which training will be stopped
        verbose=True,           # Print messages when stopping
    )

    trainer = Trainer(
        max_epochs=config.two_tower_num_epochs,
        devices=1,
        accelerator=config.device.type,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
    )

    print("🚀 DEBUG: Starting model training...")
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    train_two_tower(config)