from typing import Callable, Iterable, List, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import torch


class ViewInvarianceVisualizer():
    """Visualizes self-supervised learning data augmentation views.

    This visualizer shows the original signal and its transformed view side by side,
    helping to understand and debug SSL data augmentation pipelines. It supports
    both spectrogram and IQ data visualization.

    Args:
        data_loader (DataLoader): PyTorch DataLoader providing signal samples
        ssl_transform (Transform): SSL transformation pipeline (e.g., BYOLTransform)
        visualize_transform (callable, optional): Transform to convert data for plotting
        num_samples (int, optional): Number of samples to visualize. Defaults to 8

    Example:
        >>> visualizer = SSLViewVisualizer(
        ...     data_loader=train_loader,
        ...     ssl_transform=BYOLTransform(),
        ...     visualize_transform=complex_spectrogram_to_magnitude
        ... )
        >>> for fig in visualizer:
        ...     plt.show()
    """

    def __init__(
        self,
        data_loader,
        visualize_transform: Optional[Callable] = None,
    ):

        self.data_loader = iter(data_loader)
        self.visualize_transform = visualize_transform

    def __iter__(self) -> Iterable:
        self.data_iter = iter(self.data_loader)
        return self  # type: ignore

    def __next__(self) -> Figure:
        batch = next(self.data_iter)

        # Debugging: Print batch structure
        print(
            f"Batch received in Visualizer: {type(batch)}, Length: {len(batch)}")

        # Check if batch is a tuple and its elements
        if isinstance(batch, tuple):
            print(f"Batch contains {len(batch)} elements.")

            # Check the views structure
            print(f"Type of batch[0]: {type(batch[0])}")
            if isinstance(batch[0], list):
                print(f"Number of views: {len(batch[0])}")  # Should be 2
                # Should match expected dimensions
                print(f"View shapes: {[v.shape for v in batch[0]]}")

            # Check the labels/targets
            print(
                f"Type of batch[1]: {type(batch[1])}, Shape: {batch[1].shape if isinstance(batch[1], torch.Tensor) else 'Not a Tensor'}")

        else:
            print("Unexpected batch structure!")

        return self._visualize(batch)

    def _visualize(self, batch: Tuple[List[torch.Tensor], torch.Tensor]) -> Figure:
        """Create side-by-side visualization of all available views dynamically."""

        batch_size = len(batch[0])  # Number of samples
        views = batch[0]  # Extract all views
        num_views = len(views)  # Determine how many views are present

        print(f"Number of views in batch: {num_views}")

        # Ensure at least 2 views exist for meaningful visualization
        if num_views < 2:
            raise ValueError(f"Expected at least 2 views, but got {num_views}")

        # Create figure with dynamic columns based on available views
        fig, axes = plt.subplots(nrows=batch_size, ncols=num_views, figsize=(
            10, 5 * batch_size), frameon=True)

        # If only one sample in batch, axes will not be an array → convert to list
        if batch_size == 1:
            axes = [axes]

        # Loop through batch
        for i in range(batch_size):
            for j in range(num_views):
                view = views[j][i]  # Get the corresponding view for the sample

                # Convert to numpy if needed
                if isinstance(view, torch.Tensor):
                    view = view.numpy()

                # Ensure view is 2D
                if view.ndim == 3:
                    view = view.squeeze(0)

                # Plot view
                axes[i][j].imshow(view, aspect="auto", cmap="jet")
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])

        # Add dynamic column titles based on the number of views
        for j in range(num_views):
            axes[0][j].set_title(f"View {j + 1}")

        plt.tight_layout()
        return fig


def visualize(num_batches: int, dataloader: torch.utils.data.DataLoader):
    # Add debug prints
    print("Starting visualization...")

    # Get first batch to inspect structure
    try:
        sample_batch = next(iter(dataloader))
        print(f"Batch type: {type(sample_batch)}")
        if isinstance(sample_batch, tuple):
            print(f"Batch elements: {len(sample_batch)}")
            print(f"First element shape: {sample_batch[0].shape}")
            print(f"Second element shape: {sample_batch[1].shape}")
    except Exception as e:
        print(f"Error inspecting batch: {str(e)}")
        raise

    visualizer = ViewInvarianceVisualizer(
        data_loader=dataloader,
    )

    # Add batch counter
    batch_count = 0
    for figure in iter(visualizer):
        print(f"Processing batch {batch_count + 1}/{num_batches}")
        figure.set_size_inches(16, 9)
        plt.show()

        batch_count += 1
        if batch_count >= num_batches:
            break
