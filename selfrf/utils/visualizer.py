from typing import Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torchsig.signals.signal_lists import TorchSigSignalLists


def visualize_iq_pair(
    view1: Union[torch.Tensor, np.ndarray],
    view2: Union[torch.Tensor, np.ndarray],
    class_id: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    title_prefix: str = ""
) -> Figure:
    """
    Visualize a single pair of I/Q signals side by side.

    Args:
        view1: First view tensor of shape [2, signal_length]
        view2: Second view tensor of shape [2, signal_length]
        figsize: Figure size (width, height)
        title_prefix: Optional prefix for subplot titles

    Returns:
        Matplotlib figure with I/Q visualizations
    """
    # Convert torch tensors to numpy if needed
    if isinstance(view1, torch.Tensor):
        view1 = view1.detach().cpu().numpy()
    if isinstance(view2, torch.Tensor):
        view2 = view2.detach().cpu().numpy()

    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Extract I and Q components
    i1, q1 = view1[0], view1[1]
    i2, q2 = view2[0], view2[1]

    # Plot both I and Q on the same axes for View 1
    time_axis = np.arange(len(i1))
    axes[0].plot(time_axis, i1, label='I')
    axes[0].plot(time_axis, q1, label='Q')
    axes[0].set_title(f'{title_prefix}View 1')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot both I and Q on the same axes for View 2
    axes[1].plot(time_axis, i2, label='I')
    axes[1].plot(time_axis, q2, label='Q')
    axes[1].set_title(f'{title_prefix}View 2')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_spectrogram_pair(
    view1: Union[torch.Tensor, np.ndarray],
    view2: Union[torch.Tensor, np.ndarray],
    class_id: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = 'jet',
    title_prefix: str = ""
) -> Figure:
    """
    Visualize a single pair of spectrograms side by side.

    Args:
        view1: First view tensor of shape [height, width] or [1, height, width]
        view2: Second view tensor of shape [height, width] or [1, height, width]
        figsize: Figure size (width, height)
        cmap: Colormap for spectrograms
        title_prefix: Optional prefix for subplot titles

    Returns:
        Matplotlib figure with spectrogram visualizations
    """
    # Convert torch tensors to numpy if needed
    if isinstance(view1, torch.Tensor):
        view1 = view1.detach().cpu().numpy()
    if isinstance(view2, torch.Tensor):
        view2 = view2.detach().cpu().numpy()

    # Remove channel dimension if present
    view1 = view1.squeeze(0) if view1.ndim == 3 else view1
    view2 = view2.squeeze(0) if view2.ndim == 3 else view2

    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot spectrograms
    im1 = axes[0].imshow(view1, aspect='auto', cmap=cmap)
    im2 = axes[1].imshow(view2, aspect='auto', cmap=cmap)

    # Remove ticks
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Add sample titles
    axes[0].set_title(f'{title_prefix}View 1')
    axes[1].set_title(f'{title_prefix}View 2')

    # Add colorbars
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    if class_id is not None:
        signal_name = TorchSigSignalLists.all_signals[class_id]
        fig.suptitle(signal_name, fontsize=14)
        fig.subplots_adjust(top=0.85)

    plt.tight_layout()
    return fig


def visualize_single_sample(
    views1: Union[torch.Tensor, np.ndarray],
    views2: Union[torch.Tensor, np.ndarray],
    sample_idx: int = 0,
    mode: str = 'spectrogram',
    **kwargs
) -> Figure:
    """
    Visualize a single sample from a batch.

    Args:
        views1: Batch of first views
        views2: Batch of second views
        sample_idx: Index of sample to visualize
        mode: 'iq' or 'spectrogram'
        **kwargs: Additional arguments for visualization functions

    Returns:
        Matplotlib figure
    """
    # Extract the specified sample from the batch
    view1 = views1[sample_idx]
    view2 = views2[sample_idx]

    # Set title prefix to show sample number
    title_prefix = f"Sample {sample_idx+1} - "

    # Visualize using the appropriate pair function
    if mode == 'iq':
        return visualize_iq_pair(view1, view2, title_prefix=title_prefix, **kwargs)
    else:
        return visualize_spectrogram_pair(view1, view2, title_prefix=title_prefix, **kwargs)


def visualize_batch(batch, mode='spectrogram', max_samples=None, **kwargs):
    """
    Process a batch and visualize individual samples.

    Args:
        batch: Batch from DataLoader with format [views, labels, metadata]
        mode: 'iq' or 'spectrogram'
        max_samples: Maximum number of samples to visualize (None for all)
        **kwargs: Additional arguments for visualization functions

    Returns:
        List of figures, one per sample
    """
    views = batch[0]
    views1, views2 = views[0], views[1]
    labels = batch[1]

    batch_size = len(views1)
    if max_samples is not None:
        batch_size = min(batch_size, max_samples)

    figures = []
    for i in range(batch_size):
        fig = visualize_single_sample(
            views1, views2, sample_idx=i, mode=mode, class_id=labels[i], **kwargs)
        figures.append(fig)
