import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from selfrf.pretraining.config import TrainingConfig, print_config, parse_training_config
from tools.iq_pretraining import train


def train_on_ray(config: TrainingConfig):

    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=config,
        run_config=RunConfig(
            name="ssl_pretraining",
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    config = parse_training_config()
    print_config(config)
    train_on_ray(config)
