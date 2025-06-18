import numpy as np
import torch
from tqdm import tqdm

from selfrf.pretraining.evalutation import EvaluateKNN, VisualizeTSNE
from selfrf.pretraining.config import EvaluationConfig, parse_evaluation_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_backbone
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.models.meta_models.mlp import MLP
from selfrf.models.meta_models import ConcatMLPHead
from selfrf.data import TwoTowerTrainModule
from selfrf.pretraining.config.training_config import TrainingStage


def evaluate_fusion_model(config: EvaluationConfig):
    if not config.model_path:
        raise ValueError("model_path is required for evaluation")

    config.training_stage = TrainingStage.FUSION

    print(f"Loading fusion model from {config.model_path}")
    checkpoint = torch.load(
        config.model_path,
        map_location=config.device,
        weights_only=False,
    )

    # Build IQ encoder
    iq_model = build_backbone(config)
    iq_model.to(config.device)
    iq_model.eval()

    # Get embedding dim before building fusion head
    dummy_input = torch.randn(1, 2, config.num_iq_samples).to(config.device)
    with torch.no_grad():
        dummy_output = iq_model(dummy_input)
        if isinstance(dummy_output, tuple):
            dummy_output = dummy_output[0]
    iq_embedding_dim = dummy_output.shape[1]

    # Build metadata tower and fusion head
    meta_model = MLP(
        input_dim=config.metadata_input_dim,
        hidden_dim=config.metadata_hidden_dim,
        output_dim=config.metadata_output_dim,
    )

    fusion_head = ConcatMLPHead(
        iq_embedding_dim=iq_embedding_dim,
        meta_embedding_dim=config.metadata_output_dim,
    )

    # Wrap full model
    model = TwoTowerTrainModule(
        iq_encoder=iq_model,
        metadata_tower=meta_model,
        fusion_head=fusion_head,
        freeze_iq_encoder=True,
        freeze_metadata_tower=True
    )

    # Load checkpoint AFTER building full model
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(config.device)
    model.eval()

    # Build data
    datamodule = build_dataloader(config)
    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()

    representations = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # This part depends on your TwoTowerCollate returning correct val batches:
            (view_iq, view_mask), metadata, (indices, names) = batch
            view_iq = view_iq.to(config.device)
            metadata = metadata.to(config.device)

            z = model(view_iq, metadata)
            representations.extend(z.cpu().numpy())

            if config.dataset in {
                DatasetType.TORCHSIG_NARROWBAND,
                DatasetType.TORCHSIG_WIDEBAND
            }:
                labels.extend([get_class_list(config)[i.item()] for i in indices])
            else:
                labels.extend([str(i) for i in names])

    representations = np.array(representations)
    labels = np.array(labels)

    print(f"✅ Finished calculating fused representations: {representations.shape}")

    # t-SNE
    print("🔎 Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"tsne_fusion_plot_{model_name}.png"
    unique_labels = sorted(set(labels))
    VisualizeTSNE(x=representations, y=labels, class_list=unique_labels).visualize(save_path=plot_path)
    print(f"✅ t-SNE plot saved at {plot_path}")

    # KNN
    print("🔎 Start KNN evaluation...")
    EvaluateKNN(representations, labels, n_neighbors=config.n_neighbors).evaluate()


if __name__ == "__main__":
    config = parse_evaluation_config()
    print_config(config)
    evaluate_fusion_model(config)