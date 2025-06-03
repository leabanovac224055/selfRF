import numpy as np
import torch
from tqdm import tqdm

from selfrf.pretraining.evalutation import EvaluateKNN, VisualizeTSNE
from selfrf.pretraining.config import EvaluationConfig, parse_evaluation_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_ssl_model
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.models.meta_models.mlp import MLP
from selfrf.models.meta_models import ConcatMLPHead
from selfrf.data import TwoTowerTrainModule

def evaluate_fusion_model(config: EvaluationConfig):
    if not config.model_path:
        raise ValueError("model_path is required for evaluation")

    print(f"Loading fusion model from {config.model_path}")
    checkpoint = torch.load(
        config.model_path,
        map_location=config.device,
        weights_only=False,
    )

    # Build frozen IQ encoder
    iq_model = build_ssl_model(config).backbone
    iq_model.eval()

    # Build metadata model
    meta_model = MLP(
        input_dim=config.metadata_input_dim,
        hidden_dim=config.metadata_hidden_dim,
        output_dim=config.metadata_output_dim,
    )

    # Build fusion head
    fusion_head = ConcatMLPHead(
        iq_embedding_dim=config.fusion_iq_dim,
        meta_embedding_dim=config.metadata_output_dim,
        output_dim=config.fusion_output_dim
    )

    # Wrap everything
    model = TwoTowerTrainModule(
        iq_encoder=iq_model,
        metadata_tower=meta_model,
        fusion_head=fusion_head,
        freeze_iq_encoder=True,
        freeze_metadata_tower=True
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(config.device)
    model.eval()

    # Prepare data
    datamodule = build_dataloader(config)
    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()

    representations = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            (x_iq, _), metadata, y = batch
            x_iq = x_iq.to(config.device)
            metadata = metadata.to(config.device)

            z = model(x_iq, metadata)  # Forward fused embedding
            representations.extend(z.cpu().numpy())

            # Labels
            if config.dataset in {
                DatasetType.TORCHSIG_NARROWBAND,
                DatasetType.TORCHSIG_WIDEBAND,
                DatasetType.TWO_TOWER_NARROWBAND
            }:
                labels.extend([get_class_list(config)[i.item()] for i in y])
            else:
                labels.extend([str(i) for i in y])

    representations = np.array(representations)
    labels = np.array(labels)

    print(f"Finished calculating fused representations (shape {representations.shape})")

    print("Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"tsne_fusion_plot_{model_name}.png"
    unique_labels = sorted(set(labels))
    VisualizeTSNE(x=representations, y=labels, class_list=unique_labels).visualize(save_path=plot_path)
    print(f"t-SNE plot saved at {plot_path}")

    print("Start KNN evaluation...")
    EvaluateKNN(representations, labels, n_neighbors=config.n_neighbors).evaluate()
