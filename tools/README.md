# 🛠️ Tools — Training & Evaluation Scripts

This folder contains various utility scripts for training and evaluating models within the self-supervised learning (SSL), metadata tower, and two-tower learning pipelines.

The provided tools allow for:

- ✅ Standalone training of IQ-only SSL models.  
- ✅ Metadata tower training with contrastive learning.  
- ✅ Two-tower fusion model training.  
- ✅ Evaluation utilities for t-SNE visualizations and KNN-based representation analysis.  
- ✅ Experimental hyperparameter search for data augmentations.  

---

## 🗂️ Folder Contents

| File                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `evaluate_fusion.py`       | Evaluates a trained fusion head model using t-SNE and KNN.                  |
| `evaluate_meta_tower.py`   | Evaluates a trained metadata tower model using t-SNE and KNN.               |
| `evaluate_pretraining.py`  | Evaluates a pretrained IQ encoder using t-SNE and KNN.                      |
| `fusion_train.py`          | Trains the fusion head of a two-tower model, using frozen IQ and metadata encoders. |
| `hyperparam_tune.py`       | Experimental Optuna-based search for optimal **data augmentation parameters** for MoCo-style SSL models. *(Name may change to better reflect this.)* |
| `iq_pretraining.py`        | Runs standalone SSL pretraining for IQ signal encoders (e.g., MoCo, BYOL, DINO). |
| `meta_tower_training.py`   | Trains the metadata tower using contrastive self-supervised learning.        |

*Note: `ray_train_detection.py`, `ray_pretraining.py`, and `train_detection.py` originate from a colleague and are not actively maintained as part of the main pipeline.*

---

## 🚀 Example Workflow

1. **Train IQ encoder**  
   `python iq_pretraining.py`

2. **Train metadata tower**  
   `python meta_tower_training.py`

3. **Train fusion head (two-tower setup)**  
   `python fusion_train.py`

4. **Evaluate any model**  
   Use the appropriate evaluation script to visualize t-SNE plots or run KNN evaluation.

---

## 🛠️ Notes

- Evaluation scripts save plots and print summary metrics after completion.  
- The augmentation parameter tuning (`hyperparam_tune.py`) focuses on adjusting MoCo-style view transformations, not classic model hyperparameters.  
- Assumes datasets, feature vectors, and model checkpoints are prepared using the corresponding data preparation scripts.  

