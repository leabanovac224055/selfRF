# Utils

General utility functions and visualizations used across the project.

## Contents

- **`visualizer.py`**  
  Contains functions to visualize IQ data and spectrograms, including:
  - `visualize_iq_pair`: Plot I and Q components of two views side by side.
  - `visualize_spectrogram_pair`: Display two spectrograms side by side.
  - `visualize_single_sample`: Convenience function to visualize a single sample from a batch.
  - `visualize_batch`: Visualize an entire batch of samples.

## Notes

These utilities are primarily intended for debugging, dataset inspection, and visual monitoring of augmentations applied during training.