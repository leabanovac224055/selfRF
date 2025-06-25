import numpy as np
import torch
from tqdm import tqdm

from selfrf.pretraining.evalutation import EvaluateKNN, VisualizeTSNE
from selfrf.pretraining.config import EvaluationConfig, parse_evaluation_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_meta_model
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.config.training_config import TrainingStage

def convert_idx_to_name(idx: int, config: EvaluationConfig) -> str:
    """Convert index to either class or family name"""
    return get_class_list(config)[idx]


def evaluate_metadata_model(config: EvaluationConfig):
    print("🚀 Starting Metadata Model Evaluation...")

    if not config.model_path:
        raise ValueError("model_path is required for evaluation")
    
    config.training_stage = TrainingStage.METADATA

    # Load metadata model
    model = build_meta_model(config)

    if model is None:
        raise ValueError(f"Failed to build metadata model for config: {config.metadata_model}")

    # Load the trained model weights
    print(f"Loading weights from {config.model_path}")
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Error loading model with strict=False: {e}")
        return

    model = model.to(config.device)
    model.eval()

    representations = []
    labels = []

    # Prepare the dataloader
    datamodule = build_dataloader(config)
    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            try:
                # Print the batch structure to understand it
                #print(f"[DEBUG] Batch structure: {type(batch)}, length: {len(batch)}")

                # Unpack batch based on structure
                if isinstance(batch, tuple) and len(batch) == 2:
                    metadata, (indices, names) = batch
                elif isinstance(batch, tuple) and len(batch) == 3:
                    metadata, iq_data, (indices, names) = batch
                elif isinstance(batch, list) and len(batch) == 2:
                    metadata, labels = batch
                    indices, names = labels, labels  # Assume labels are directly returned
                else:
                    print(f"[ERROR] Unexpected batch format: {batch}")
                    continue

                metadata = metadata.to(config.device)

                # Forward pass through the metadata model
                z = model(metadata)

                # Append representations
                representations.extend(z.cpu().numpy())

                # Append labels
                safe_names = [str(n) for n in indices]
                labels.extend(safe_names)

            except Exception as e:
                #print(f"[ERROR] Issue processing batch: {e}")
                continue

    # Check for empty representations
    if len(representations) == 0:
        print("[ERROR] No representations were gathered during evaluation. Check the batch processing logic.")
        return

    representations = np.array(representations)
    labels = np.array(labels)
    print(f"Finished calculating representations (shape {representations.shape})")

    # Check if enough samples for t-SNE
    if len(representations) < config.n_neighbors:
        print(f"[ERROR] Not enough samples ({len(representations)}) to perform t-SNE with perplexity {config.n_neighbors}.")
        return

    print("Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"tsne_plot_{model_name}.png"

    # Use unique labels for visualization
    unique_labels = sorted(set(labels))
    VisualizeTSNE(
        x=representations,
        y=labels,
        class_list=unique_labels,
    ).visualize(save_path=plot_path)
    print(f"t-SNE plot saved at {plot_path}")

    print("Start KNN evaluation...")
    EvaluateKNN(representations, labels, n_neighbors=config.n_neighbors).evaluate()


if __name__ == "__main__":
    config = parse_evaluation_config()
    print_config(config)
    evaluate_metadata_model(config)