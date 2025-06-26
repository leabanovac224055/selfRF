# 🛠️ Tools — Training & Evaluation Scripts

This folder contains various utility scripts for training and evaluating models within the self-supervised learning (SSL), metadata tower, and two-tower learning pipelines.

The provided tools allow for:

- ✅ Standalone training of IQ-only SSL models.  
- ✅ Metadata tower training with contrastive learning.  
- ✅ Two-tower fusion model training (requires pretrained IQ and metadata models).  
- ✅ Evaluation utilities for t-SNE visualizations and KNN-based representation analysis.  
- ✅ Experimental hyperparameter search for data augmentations.  

---

## 🗂️ Folder Contents

| File                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `evaluate_fusion.py`       | Evaluates a trained fusion head model using t-SNE and KNN.                  |
| `evaluate_meta_tower.py`   | Evaluates a trained metadata tower model using t-SNE and KNN.               |
| `evaluate_pretraining.py`  | Evaluates a pretrained IQ encoder using t-SNE and KNN.                      |
| `fusion_train.py`          | Trains the fusion head of a two-tower model using pretrained IQ and metadata models. The pretrained model paths must be manually inserted into the script. Freezing the encoders is optional. |
| `hyperparam_tune.py`       | Experimental Optuna-based search for optimal **data augmentation parameters** for MoCo-style SSL models. *(Name may change to better reflect this.)* |
| `iq_pretraining.py`        | Runs standalone SSL pretraining for IQ signal encoders (e.g., MoCo, BYOL, DINO). |
| `meta_tower_training.py`   | Trains the metadata tower using contrastive self-supervised learning.        |


---

## 🛠️ Notes

- Evaluation scripts save plots and print summary metrics after completion.  
- The augmentation parameter tuning (`hyperparam_tune.py`) focuses on adjusting MoCo-style view transformations, not classic model hyperparameters.  
- Assumes datasets, feature vectors, and model checkpoints are prepared using the corresponding data preparation scripts.  
- Example commands for running these scripts can be found in the [main repository README](../README.md).

