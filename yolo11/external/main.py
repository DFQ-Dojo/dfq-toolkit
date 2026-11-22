from ultralytics.cfg import TASK2DATA
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    checks,
    yaml_load,
)


from external.yolo import YOLO
from external.config import get_config, TrainConfig


# this is copy verbatim except for its usage of YOLO, which is a modified version in ./yolo.py
def train(
    self: YOLO,
    trainer=None,
    quant_mode=None,
    gimg=None,
    calibration=1000,
    kd=None,
    ptq=False,
    **kwargs,
):
    """
    Trains the model using the specified dataset and training configuration.

    This method facilitates model training with a range of customizable settings. It supports training with a
    custom trainer or the default training approach. The method handles scenarios such as resuming training
    from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

    When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
    arguments and warns if local arguments are provided. It checks for pip updates and combines default
    configurations, method-specific defaults, and user-provided arguments to configure the training process.

    Args:
        trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
        **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
            data (str): Path to dataset configuration file.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            imgsz (int): Input image size.
            device (str): Device to run training on (e.g., 'cuda', 'cpu').
            workers (int): Number of worker threads for data loading.
            optimizer (str): Optimizer to use for training.
            lr0 (float): Initial learning rate.
            patience (int): Epochs to wait for no observable improvement for early stopping of training.

    Returns:
        (Dict | None): Training metrics if available and training is successful; otherwise, None.

    Raises:
        AssertionError: If the model is not a PyTorch model.
        PermissionError: If there is a permission issue with the HUB session.
        ModuleNotFoundError: If the HUB SDK is not installed.

    Examples:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.train(data="coco8.yaml", epochs=3)
    """
    self._check_is_pytorch_model()
    if (
        hasattr(self.session, "model") and self.session.model.id
    ):  # Ultralytics HUB session with loaded model
        if any(kwargs):
            LOGGER.warning(
                "WARNING ⚠️ using HUB training arguments, ignoring local training arguments."
            )
        kwargs = self.session.train_args  # overwrite kwargs

    checks.check_pip_update_available()

    overrides = (
        yaml_load(checks.check_yaml(kwargs["cfg"]))
        if kwargs.get("cfg")
        else self.overrides
    )
    custom = {
        # NOTE: handle the case when 'cfg' includes 'data'.
        "data": overrides.get("data")
        or DEFAULT_CFG_DICT["data"]
        or TASK2DATA[self.task],
        "model": self.overrides["model"],
        "task": self.task,
    }  # method defaults
    args = {
        **overrides,
        **custom,
        **kwargs,
        "mode": "train",
    }  # highest priority args on the right
    if args.get("resume"):
        args["resume"] = self.ckpt_path

    self.trainer = (trainer or self._smart_load("trainer"))(
        overrides=args, _callbacks=self.callbacks
    )

    if gimg is not None:
        self.trainer.use_gimg(gimg, calibration)

    if kd is not None:
        self.trainer.use_kd(kd)

    if ptq:
        self.trainer.use_ptq()

    if not args.get("resume"):  # manually set model only if not resuming
        self.trainer.model = self.trainer.get_model(
            weights=self.model if self.ckpt else None, cfg=self.model.yaml
        )
        self.model = self.trainer.model

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train(quant_mode=quant_mode)
    # Update model and cfg after training
    if RANK in {-1, 0}:
        ckpt = (
            self.trainer.best
            if self.trainer.best.exists()
            else self.trainer.last
        )
        self.model, self.ckpt = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(
            self.trainer.validator, "metrics", None
        )  # TODO: no metrics returned by DDP
    return self.metrics


def original_train(config: TrainConfig):
    """Start train the model.

    Parameters
    ----------
    config : TrainConfig
        Please refer to the doc of TrainConfig.
    """

    model = YOLO(config.model)

    train(
        model,
        device=config.device[0] if len(config.device) == 1 else config.device,
        data=config.dataset_manifest,
        batch=config.batch_size,
        fraction=config.fraction,
        gimg=config.generated_weights_path,
        calibration=config.calibration_size,
        quant_mode=config.model_quantize_mode,
        epochs=config.end_epochs,
        patience=config.patience,
        optimizer=config.hyps.optimizer_name,
        lr0=config.hyps.lr0,
        lrf=config.hyps.lrf,
        momentum=config.hyps.momentum,
        weight_decay=config.hyps.weight_decay,
        box=config.hyps.box,
        cls=config.hyps.cls,
        dfl=config.hyps.dfl,
        kd=config.kd_method,
        ptq=config.ptq,
    )


if __name__ == "__main__":
    config = get_config()
    LOGGER.info(config.model_dump_json(indent=2))
    original_train(config)
