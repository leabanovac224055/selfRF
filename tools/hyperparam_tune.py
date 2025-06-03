import os
from dotenv import load_dotenv
import torch
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import optuna.visualization 
from torch.utils.data import Subset
import random
import numpy as np

from selfrf.pretraining.config import parse_training_config
from selfrf.pretraining.factories import build_dataloader, build_ssl_model
from selfrf.pretraining.utils.callbacks import ModelAndBackboneCheckpoint
from selfrf.pretraining.utils.enums import SSLModelType
from selfrf.transforms import MoCoTransform

def train(config, moco_transform, trial):
    # Phase 1: SSL backbone training
    datamodule = build_dataloader(config)
    datamodule.prepare_data()
    datamodule.setup()
    
    subset_fraction = 0.1  # ✅ Use 10% of training dataset
    dataset = datamodule.train_dataset
    total = len(dataset)
    n = int(subset_fraction * total)
    random.seed(42)
    indices = random.sample(range(total), n)
    datamodule.train_dataset = Subset(dataset, indices)
    print(f"✅ Using a subset of {n}/{total} training samples")

    # Inject the dynamic transform into the datamodule
    if hasattr(datamodule, 'train_dataset'):
        datamodule.train_dataset.transform = moco_transform
    elif hasattr(datamodule, 'train_transform'):
        datamodule.train_transform = moco_transform
    else:
        print("⚠️ Warning: Unable to inject transform into the datamodule.")

    ssl_model = build_ssl_model(config)

    # Logger with TensorBoard
    logger = TensorBoardLogger(
        save_dir=config.training_path,
        name=f"moco_tuning/trial_{trial.number}",
        default_hp_metric=False
    )

    # Checkpoint saver
    checkpoint_callback = ModelAndBackboneCheckpoint(
        dirpath=logger.log_dir,  # Automatically resolves to: moco_tuning/trial_{trial.number}
        filename=f"moco-{config.backbone.value}-{config.dataset.value}-bs{config.batch_size}-e{{epoch:d}}-loss{{train_loss:.3f}}",
        save_top_k=5,
        verbose=True,
        save_last=True,
        monitor="val_online_cls_loss",
        mode="min",
    )
    
    early_stopping = EarlyStopping(
        monitor="val_online_cls_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    trainer = Trainer(
        precision="16-mixed",
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=config.device.type,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
    )

    # Training
    trainer.fit(model=ssl_model, datamodule=datamodule)
    
    import json
    with open(f"trial_{trial.number}_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in trainer.callback_metrics.items()}, f, indent=2)


    # Get the final validation loss from the trainer's callback metrics
    final_loss = trainer.callback_metrics.get("val_online_cls_loss")
    if final_loss is not None:
        print(f"🚀 Final Validation Loss for Trial {trial.number}: {final_loss.item()}")
        return final_loss.item()
    else:
        print("⚠️ Warning: Training loss not available, returning infinity.")
        return float("inf")
    

def objective(trial):
    config = parse_training_config()
    config.ssl_model = SSLModelType.MOCOV3

    # Strong view (View 1)
    view_1_params = {
        "max_time_shift": trial.suggest_int("view1_max_time_shift", 700, 1000),
        "max_freq_shift": trial.suggest_float("view1_max_freq_shift", 0.2, 0.5),
        "tr_prob": trial.suggest_float("view1_tr_prob", 0.6, 0.8),
        "si_prob": trial.suggest_float("view1_si_prob", 0.6, 0.8),
        "noise_power_db": (
            trial.suggest_float("view1_noise_min", -90, -70),
            trial.suggest_float("view1_noise_max", -70, -50),
        ),
        "cutout_duration": (
            trial.suggest_float("view1_cutout_min", 0.03, 0.08),
            trial.suggest_float("view1_cutout_max", 0.08, 0.2),
        ),
        "min_amplitude_scale": trial.suggest_float("view1_amp_min", 0.3, 0.8),
        "max_amplitude_scale": trial.suggest_float("view1_amp_max", 1.5, 3.0),
        "max_phase_shift_rad": trial.suggest_float("view1_phase_shift", np.pi / 6, np.pi / 2),
    }

    # Weak view (View 2)
    view_2_params = {
        "max_time_shift": trial.suggest_int("view2_max_time_shift", 100, 500),
        "max_freq_shift": trial.suggest_float("view2_max_freq_shift", 0.05, 0.3),
        "tr_prob": trial.suggest_float("view2_tr_prob", 0.3, 0.6),
        "si_prob": trial.suggest_float("view2_si_prob", 0.3, 0.6),
        "noise_power_db": (
            trial.suggest_float("view2_noise_min", -110, -90),
            trial.suggest_float("view2_noise_max", -90, -70),
        ),
        "cutout_duration": (
            trial.suggest_float("view2_cutout_min", 0.01, 0.06),
            trial.suggest_float("view2_cutout_max", 0.06, 0.15),
        ),
        "min_amplitude_scale": trial.suggest_float("view2_amp_min", 0.6, 0.9),
        "max_amplitude_scale": trial.suggest_float("view2_amp_max", 1.2, 2.0),
        "max_phase_shift_rad": trial.suggest_float("view2_phase_shift", np.pi / 8, np.pi / 3),
    }

    # ✅ Create the rich transform with all parameters
    moco_transform = MoCoTransform(
        view_1_params=view_1_params,
        view_2_params=view_2_params,
    )

    return train(config, moco_transform, trial)

if __name__ == '__main__':
    load_dotenv()

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize", 
        study_name="moco_tuning_27_05", 
        storage="sqlite:///moco_tuning_27_05.db", 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=7)

    # Save best trial and metrics
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (train_loss): {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # Save the study to file for later analysis
    study.trials_dataframe().to_csv("moco_tuning_results2.csv")
    
# This is with train_loss
#Best is trial 6 with value: 0.29449462890625.
#Best trial: 6
#Best value (train_loss): 0.29449462890625
#Best parameters: {'view1_max_time_shift': 805, 'view1_max_freq_shift': 0.3985574086980508, 'view1_tr_prob': 0.608450454556431, 'view1_si_prob': 0.7762550830719575, 'view1_noise_min': -75.22313643070225, 'view1_noise_max': -62.118486958653406, 'view1_cutout_min': 0.04555615917235689, 'view1_cutout_max': 0.17060771871824668, 'view1_amp_min': 0.7624829965779709, 'view1_amp_max': 2.0610869794590556, 'view1_phase_shift': 1.1226980361020837, 
# 'view2_max_time_shift': 387, 'view2_max_freq_shift': 0.18816041163381997, 'view2_tr_prob': 0.4608719907109846, 'view2_si_prob': 0.43061635232409606, 'view2_noise_min': -101.52797781738239, 'view2_noise_max': -73.12024758794661, 'view2_cutout_min': 0.028072889881628003, 'view2_cutout_max': 0.06292169619335006, 'view2_amp_min': 0.7678840019601538, 'view2_amp_max': 1.742623699317833, 'view2_phase_shift': 0.9994746248630416}

# This is with val_online_cls_loss
#Best is trial 1 with value: 3.2614526748657227.
#Best trial: 1
#Best value (train_loss): 3.2614526748657227
#Best parameters: {'view1_max_time_shift': 825, 'view1_max_freq_shift': 0.4869832744847659, 'view1_tr_prob': 0.6335088148702812, 'view1_si_prob': 0.760995489803916, 'view1_noise_min': -79.78656282524992, 'view1_noise_max': -50.569569307756105, 'view1_cutout_min': 0.06209927873433858, 'view1_cutout_max': 0.19842891172007926, 'view1_amp_min': 0.4451500335796409, 'view1_amp_max': 2.4424010981656092, 'view1_phase_shift': 0.8290366337026638, 'view2_max_time_shift': 317, 'view2_max_freq_shift': 0.22119178046449245, 'view2_tr_prob': 0.32602448100031234, 'view2_si_prob': 0.4225862707365518, 'view2_noise_min': -98.91180607062145, 'view2_noise_max': -73.86129199301135, 'view2_cutout_min': 0.014173617169049677, 'view2_cutout_max': 0.09737721020374648, 'view2_amp_min': 0.758883323818787, 'view2_amp_max': 1.7668596653675825, 'view2_phase_shift': 0.743366436618919}