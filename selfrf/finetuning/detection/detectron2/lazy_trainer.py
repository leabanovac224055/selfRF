
import logging
import torch
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_writers,
    hooks,
)
from detectron2.data import (
    build_detection_train_loader,
    get_detection_dataset_dicts,
)

from selfrf.finetuning.detection.detectron2.config import Detectron2Config
from selfrf.finetuning.detection.detectron2.model_conf.builds import get_build_functions

from .mapper import mapper


def do_test(config, model):
    """
    Run evaluation on the validation dataset
    """
    # Create evaluator for your validation dataset
    evaluator = COCOEvaluator(
        dataset_name="torchsig_wideband_val",
        output_dir=config.output_dir,
        tasks=("bbox",),  # Only evaluate bounding boxes
        use_fast_impl=True
    )

    # Build test loader using same mapper as training
    val_loader = build_detection_test_loader(
        dataset=get_detection_dataset_dicts(
            names="torchsig_wideband_val",
            filter_empty=False,
        ),
        mapper=mapper,
        num_workers=2
    )

    # Run evaluation
    results = inference_on_dataset(model, val_loader, evaluator)

    # Print results
    print_csv_format(results)
    # Return results for logging
    return results


def do_train_lazy(config: Detectron2Config):

    build_model, build_optimizer, build_lr, build_training = get_build_functions(
        config)
    model_config = build_model()
    model = instantiate(model_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))

    optimizer_config = build_optimizer()
    optimizer_config.params.model = model
    optim = instantiate(optimizer_config)

    # train_loader = instantiate(cfg.dataloader.train)

    train_loader = build_detection_train_loader(
        mapper=mapper,
        dataset=get_detection_dataset_dicts(
            names="torchsig_wideband_train",
            filter_empty=False,  # Keep all spectrograms
        ),
        total_batch_size=config.ims_per_batch,
    )

    train_config = build_training(config)

    trainer = (AMPTrainer if train_config.amp.enabled else SimpleTrainer)(
        model, train_loader, optim)

    checkpointer = DetectionCheckpointer(
        model,
        train_config.output_dir,
        trainer=trainer,
    )

    lr_scheduler_config = build_lr(config)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(lr_scheduler_config)),
            (
                hooks.PeriodicCheckpointer(
                    checkpointer, **train_config.checkpointer)
            ),
            hooks.EvalHook(train_config.eval_period,
                           lambda: do_test(train_config, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(train_config.output_dir,
                                    train_config.max_iter),
                    period=train_config.log_period,
                )

            ),
        ]
    )

    # run the training
    checkpointer.resume_or_load(path="", resume=False)
    trainer.train(0, train_config.max_iter)

    # final evaluation
    final_results = do_test(train_config, model)
    logger.info("Final evaluation results:")
    print_csv_format(final_results)
