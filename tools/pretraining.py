import os
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.factories import build_dataset, build_ssl_model
from selfrf.pretraining.utils.callbacks import ModelAndBackboneCheckpoint


def train(config: TrainingConfig):

    datamodule = build_dataset(config)
    datamodule.prepare_data()
    datamodule.setup()

    if not config.online_linear_eval:
        datamodule.val_dataloader = None

    ssl_model = build_ssl_model(config)

    # Configure TensorBoardLogger
    logger = TensorBoardLogger(
        os.path.join(config.training_path, config.ssl_model.value)
    )

    # Configure ModelCheckpoint
    checkpoint_callback = ModelAndBackboneCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            f"{config.ssl_model.value}"
            f"-{config.backbone.value}"
            f"-{config.dataset.value}"
            f"-{'spec' if config.spectrogram else 'iq'}"
            f"-e{{epoch:d}}"
            f"-b{config.batch_size}"
            f"-loss{{train_loss:.3f}}"
        ),
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        # callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model=ssl_model, datamodule=datamodule)


if __name__ == '__main__':
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    train(config)
