from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class EvaluateKNN():
    """Evaluate features using K-Nearest Neighbors classifier.

    This evaluation method is commonly used to assess the quality of 
    learned representations from self-supervised models.
    """

    def __init__(self,
                 x: List[np.ndarray],
                 y: List[str],
                 split: float = 0.8,
                 shuffle: bool = True,
                 n_neighbors: int = 50,
                 verbose: bool = True):
        """Initialize KNN evaluation.

        Args:
            x: Feature vectors to evaluate
            y: Corresponding labels
            split: Train/test split ratio
            shuffle: Whether to shuffle data before splitting
            n_neighbors: Number of neighbors for KNN
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose

        if self.verbose:
            print(f"Evaluating with KNN (k={n_neighbors})")
            print(f"Total samples: {len(x)}")
            # Count unique classes
            unique_classes = set(y)
            print(f"Number of classes: {len(unique_classes)}")

        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, train_size=split, shuffle=shuffle, random_state=42)

        if self.verbose:
            print(f"Train set: {len(self.x_train)} samples")
            print(f"Test set: {len(self.x_test)} samples")

        self.n_neighbors = n_neighbors

    def evaluate(self) -> float:
        """Evaluate features with KNN classifier.

        Returns:
            Classification accuracy on test set
        """
        if self.verbose:
            print("Fitting KNN classifier...")

        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Fit the model
        knn.fit(self.x_train, self.y_train)

        if self.verbose:
            print("Predicting on test set...")

        # Predict the labels
        pred_labels = knn.predict(self.x_test)

        # Calculate accuracy
        acc = accuracy_score(self.y_test, pred_labels)

        if self.verbose:
            print(f"KNN Accuracy: {acc:.4f}")

        return acc
