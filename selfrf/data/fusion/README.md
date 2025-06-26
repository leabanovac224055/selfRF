# 📁 Fusion Head & Two-Tower Training Utilities

This folder contains the modules required to integrate and train the **fusion head** that combines IQ data and metadata embeddings within a two-tower learning architecture.

The provided tools allow for:

- ✅ Fusion of IQ and metadata embeddings.  
- ✅ Contrastive self-supervised learning on the fused representations.  
- ✅ Flexible freezing of IQ encoder and metadata tower during training.  

---

## 🗂️ Folder Contents

| File                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `fusion_module.py`          | Defines a modular `FusionModel` that fuses IQ and metadata embeddings using a fusion head. |
| `two_tower_train_module.py` | Provides a PyTorch Lightning module for training the two-tower model with contrastive loss. Supports flexible freezing and detailed input checks. |

---

## 🔗 Fusion Model (`fusion_module.py`)

The `FusionModel` wraps:

- A frozen **IQ backbone** (e.g., pretrained encoder for IQ signals).  
- A trained or trainable **metadata tower** for metadata feature vectors.  
- A **fusion head**, the only trainable component by default, which merges IQ and metadata embeddings into a final representation.

This architecture allows:

- Efficient reuse of frozen IQ and metadata encoders.  
- Focused training of the fusion mechanism for combined representation learning.  

---

## 🧠 Two-Tower Training Module (`two_tower_train_module.py`)

The `TwoTowerTrainModule` implements contrastive self-supervised learning for the combined IQ and metadata embeddings.

**Key Features:**

- Optionally freezes IQ encoder and/or metadata tower.  
- Applies NT-Xent contrastive loss on fused embeddings.  
- Provides detailed sanity checks for NaNs and invalid inputs.  
- Logs training loss during the fusion head training phase.  
- Supports unpacking of `(cls_token, pooled_token)` style encoder outputs.  

---

## 🛠️ Notes

- Assumes preprocessed `.zarr` IQ data and scaled metadata feature vectors are available.  
- IQ encoder and metadata tower are typically pretrained separately before fusion head training.  
- The fusion head is expected to be a lightweight MLP or similar module for embedding combination.  