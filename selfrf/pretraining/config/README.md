
# ⚙️ Configuration Utilities for Training & Evaluation

This folder contains dataclass-based configuration utilities for defining and parsing parameters across different training and evaluation pipelines.

The goal is to provide:

- ✅ Reusable, structured config classes for both training and evaluation.  
- ✅ Clean CLI argument parsing using `argparse`.  
- ✅ A single source of truth for default hyperparameters and dataset settings.  
- ✅ Easy overrides of parameters via command-line arguments.  

---

## 🗂️ Folder Contents

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `__init__.py`             | Exposes common config classes and parsing functions for external use.       |
| `base_config.py`          | Defines `BaseConfig`, containing shared parameters for datasets, backbone, device, and metadata tower. Includes CLI parsing logic. |
| `evaluation_config.py`    | Extends `BaseConfig` for evaluation-specific parameters (e.g., model paths, t-SNE, KNN). |
| `training_config.py`      | Extends `BaseConfig` for training-specific parameters (e.g., SSL model type, epochs, training stage). |

---

## 🧩 How it Works

All config classes inherit from `BaseConfig`, which provides core fields such as:

- Dataset selection  
- Batch size and worker settings  
- Backbone architecture and provider  
- Metadata tower input/output dimensions  
- Device management  

`training_config.py` adds parameters specific to model training:  

- SSL model selection (e.g., BYOL, MoCo)  
- Training stage (SSL, metadata tower, fusion head)  
- MoCo-v3 hyperparameters (projection dim, momentum, etc.)  

`evaluation_config.py` adds parameters for evaluation:  

- Model checkpoint path  
- t-SNE and KNN options  
- Evaluation output folder  

---

## 🛠️ Example Usage in Scripts

**Training script:**

```python
from selfrf.pretraining.config import parse_training_config

config = parse_training_config()
print(config)
```

**Evaluation script:**

```python
from selfrf.pretraining.config import parse_evaluation_config

config = parse_evaluation_config()
print(config)
```

These functions automatically parse CLI arguments and populate the appropriate config objects.

---

## 🗃️ Notes

- Default values are defined as constants in each config file for easy maintenance.  
- CLI argument choices (e.g., for dataset type or SSL model) are enforced using enums.  
- The structure ensures consistency across your pipelines while still allowing flexibility.  