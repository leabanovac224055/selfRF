
# Metadata Tower and Fusion Head Models

This folder contains simple neural network modules used in the **metadata tower** and **fusion head** of the two-tower architecture for RF signal learning.

## 📦 Contents

| File               | Description                                           |
|--------------------|-------------------------------------------------------|
| `mlp.py`           | MLP model for processing structured metadata vectors. |
| `concat_mlp_head.py` | Fusion head that combines IQ and metadata embeddings via concatenation and an MLP. |
| `__init__.py`      | Module initialization for easy imports.               |

---

## 🧠 Model Descriptions

### `MLP`
- Processes structured metadata feature vectors such as:
  - Center Frequency
  - Bandwidth
  - Duration
- Produces a normalized embedding suitable for use with contrastive losses (e.g., NT-Xent).

**Usage Example:**

```python
from selfrf.models.meta_models import MLP

model = MLP(input_dim=3, hidden_dim=128, output_dim=128)
output = model(metadata_tensor)  # metadata_tensor: shape [B, 3]
```

---

### `ConcatMLPHead`
- Combines IQ data embeddings and metadata embeddings into a fused representation.
- Implements a simple MLP with dropout and batch normalization.

**Usage Example:**

```python
from selfrf.models.meta_models import ConcatMLPHead

model = ConcatMLPHead(iq_embedding_dim=256, meta_embedding_dim=64)
fused = model(iq_embedding, meta_embedding)
```

---

## ✅ Notes
- Both models use **batch normalization** for training stability.
- The MLP normalizes outputs to unit length, compatible with cosine similarity objectives.
- Fusion head expects pre-computed IQ and metadata embeddings.