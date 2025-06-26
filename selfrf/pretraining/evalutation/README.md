# 📊 Evaluation Utilities (`selfrf/pretraining/evalutation`)

This folder contains utilities for evaluating learned representations from self-supervised models. It provides tools for:

- t-SNE visualization to inspect clustering of learned features.
- K-Nearest Neighbors (KNN) evaluation to quantify representation quality.

---

## 📁 Folder Structure

```
selfrf/pretraining/evalutation/
├── __init__.py         # Import shortcuts for evaluation tools
├── knn.py              # KNN-based representation evaluation
└── tsne.py             # t-SNE visualization
```

---

## 🧩 Module Descriptions

### `__init__.py`
Convenience imports:
```python
from .knn import EvaluateKNN
from .tsne import VisualizeTSNE
```

---

### `knn.py` — K-Nearest Neighbors Evaluation

Class: `EvaluateKNN`

Performs a simple supervised classification using K-Nearest Neighbors to evaluate the structure of learned representations.

**Arguments:**
- `x` (*List[np.ndarray]*): Feature vectors.
- `y` (*List[str]*): Corresponding class labels.
- `split` (*float*, default=0.8): Train/test split ratio.
- `shuffle` (*bool*, default=True): Whether to shuffle before splitting.
- `n_neighbors` (*int*, default=50): Number of neighbors for KNN.
- `verbose` (*bool*, default=True): Verbosity flag.

**Example:**
```python
evaluator = EvaluateKNN(x=features, y=labels, n_neighbors=50)
acc = evaluator.evaluate()
print(f"KNN Accuracy: {acc:.4f}")
```

---

### `tsne.py` — t-SNE Visualization

Class: `VisualizeTSNE`

Visualizes high-dimensional learned representations in 2D using t-SNE.

**Arguments:**
- `x` (*List[np.ndarray]*): Feature vectors.
- `y` (*List[str]*): Class labels.
- `class_list` (*List[str]*): Full list of class names (for consistent coloring).

**Example:**
```python
visualizer = VisualizeTSNE(x=features, y=labels, class_list=all_classes)
visualizer.visualize(save_path="tsne_plot.png")
```

---

## 📝 Notes
- Both tools expect feature vectors already extracted from trained models.
- t-SNE provides qualitative insights into clustering, while KNN offers a simple quantitative benchmark.
- These tools are designed to evaluate both IQ signal and metadata-based representations.