import numpy as np
import torch
from tqdm import tqdm

from selfrf.pretraining.evalutation import EvaluateKNN, VisualizeTSNE
from selfrf.pretraining.config import EvaluationConfig, parse_evaluation_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_backbone
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import DatasetType


def convert_idx_to_name(idx: int, config: EvaluationConfig) -> str:
    """Convert index to either class or family name"""
    return get_class_list(config)[idx]


def evaluate(config: EvaluationConfig):

    if not config.model_path:
        raise ValueError("model_path is required for evaluation")

    datamodule = build_dataloader(config)

    model = build_backbone(config)

    if config.model_path.lower() == "random" or not config.model_path:
        print("Using randomly initialized weights")
        # Model already has random weights from initialization
    else:
        print(f"Loading weights from {config.model_path}")
        checkpoint = torch.load(
            config.model_path,
            map_location=config.device,
            weights_only=False,
        )

        # Handle state_dict format if needed
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Load the weights
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Failed to load weights exactly: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint, strict=False)

    model = model.to(config.device)
    model.eval()

    representations = []
    labels = []

    datamodule.setup("fit")
    val_dataloader = datamodule.train_dataloader()
    with torch.no_grad():  # No gradient needed

        for x, (indices, names) in tqdm(val_dataloader):
            x = x.to(config.device)

            z = model(x)

            representations.extend(z.cpu().numpy())

            if config.dataset in {
                DatasetType.TORCHSIG_NARROWBAND,
                DatasetType.TORCHSIG_WIDEBAND
            }:
                labels.extend([get_class_list(config)[i.item()] for i in indices])
            else:
                # Use real class names (and make sure they're plain strings)
                safe_names = [
                    n.item() if isinstance(n, np.ndarray) else str(n) for n in names
                ]
                labels.extend(safe_names)

    representations = np.array(representations)
    labels = np.array(labels)
    print(
        f"Finished calculating representations (shape {representations.shape})")

    print("Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"tsne_plot_{model_name}.png"
    
    # ✅ Use real class names
    unique_labels = sorted(set(labels))
    VisualizeTSNE(
        x=representations,
        y=labels,
        class_list=unique_labels,
    ).visualize(save_path=plot_path)
    print(f"t-SNE plot saved at {plot_path}")

    print("Start KNN evaluation...")
    EvaluateKNN(representations, labels,
                n_neighbors=config.n_neighbors).evaluate()


if __name__ == "__main__":
    config = parse_evaluation_config()
    print_config(config)
    evaluate(config)
