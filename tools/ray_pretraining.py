from datetime import datetime, time
import os
import pyarrow
import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from selfrf.pretraining.config import TrainingConfig, print_config, parse_training_config
from pretraining import train


def train_on_ray(config: TrainingConfig):

    ray.init()

    fs = pyarrow.fs.S3FileSystem(
        endpoint_override=os.environ['MINIO_ENDPOINT'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
    )

    # Generate a unique timestamp with both date and time components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use the timestamp in your job name
    job_name = f"ssl_pretraining_{timestamp}"

    print(f"Starting Ray training job: {job_name}")
    trainer = TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=config,
        run_config=RunConfig(
            # Use time.time() for unique name
            name=job_name,
            storage_filesystem=fs,
            storage_path="iqdm-ai/training",
        ),
        scaling_config=ScalingConfig(
            num_workers=4,
            use_gpu=True,
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    config = parse_training_config()
    print_config(config)
    train_on_ray(config)
