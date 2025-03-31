import argparse
from datetime import datetime
import os
import warnings
import pyarrow
import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, build_detectron2_config, print_config
from train_detection import train


def train_on_ray(config: Detectron2Config):
    warnings.filterwarnings("ignore")

    # Generate a unique timestamp for the job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"detectron2_training_{timestamp}"

    ray.init()

    fs = pyarrow.fs.S3FileSystem(
        endpoint_override=os.environ['MINIO_ENDPOINT'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=config,
        run_config=RunConfig(
            name=job_name,
            storage_filesystem=fs,
            storage_path="iqdm-ai/training",
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train_on_ray(config)
