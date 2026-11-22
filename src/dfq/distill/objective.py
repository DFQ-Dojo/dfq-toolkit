"""The distillation objective of quantization-aware training.

The student is the quantized detector; the teacher is the same detector in full precision. One
step computes three terms on the same images:

- **detection:** the student's own training losses, summed;
- **logit:** a KL divergence or DIST loss between the outputs of each model's `logit_layer`;
- **feature:** the mean squared difference between the outputs of the `feature_layers`.

The teacher runs `predict` in evaluation mode without gradients, so its dropout and drop-path are
off and its normalization statistics stay fixed.
"""

from collections.abc import Sequence
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from dfq.detection import Detector, Targets
from dfq.detection.taps import tap
from dfq.distill.losses import SoftmaxOver, dist_loss, feature_mse, kl_divergence


class KLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["kl"] = "kl"
    tau: float = 1.0
    softmax_over: SoftmaxOver = SoftmaxOver.EACH_OUTPUT


class DISTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dist"] = "dist"
    beta: float = 1.0
    gamma: float = 1.0
    tau: float = 1.0


class DistillConfig(BaseModel):
    """The weights of the three terms, the logit loss, and the kind of feature layers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_weight: float = 1.0
    logit_weight: float = 1.0
    feature_weight: float = 1.0
    logit_loss: Annotated[KLConfig | DISTConfig, Field(discriminator="kind")] = KLConfig()
    feature_kind: str = Field(description="passed to Detector.feature_layers, such as 'block'")


def logit_loss(
    student: list[Tensor], teacher: list[Tensor], config: KLConfig | DISTConfig
) -> Tensor:
    if isinstance(config, KLConfig):
        return kl_divergence(student, teacher, config.tau, config.softmax_over)
    return dist_loss(student, teacher, config.beta, config.gamma, config.tau)


def tensors_of(outputs: dict[str, object], names: list[str]) -> list[Tensor]:
    tensors = []
    for name in names:
        output = outputs[name]
        if not isinstance(output, Tensor):
            raise TypeError(f"feature layer {name} returned {type(output).__name__}, not a tensor")
        tensors.append(output)
    return tensors


def teacher_outputs(teacher: Detector, images: Tensor, names: list[str]) -> dict[str, object]:
    teacher.network.eval()
    with torch.no_grad(), tap(teacher.network, names) as outputs:
        teacher.predict(images)
    return outputs


def distill_losses(
    student: Detector,
    teacher: Detector,
    images: Tensor,
    targets: Sequence[Targets],
    config: DistillConfig,
) -> dict[str, Tensor]:
    """Return the weighted terms and their sum, `"loss"`. The student must be in training mode."""
    logit_name = student.logit_layer()
    feature_names = student.feature_layers(config.feature_kind)
    names = [logit_name, *feature_names]
    from_teacher = teacher_outputs(teacher, images, names)
    with tap(student.network, names) as from_student:
        detection = student.losses(images, targets)
    detection_loss = torch.stack(list(detection.values())).sum()
    student_logits = student.logits(from_student[logit_name])
    teacher_logits = teacher.logits(from_teacher[logit_name])
    logit = logit_loss(student_logits, teacher_logits, config.logit_loss)
    terms = {"detection": config.detection_weight * detection_loss}
    terms["logit"] = config.logit_weight * logit
    if feature_names:
        student_features = tensors_of(from_student, feature_names)
        teacher_features = tensors_of(from_teacher, feature_names)
        terms["feature"] = config.feature_weight * feature_mse(student_features, teacher_features)
    terms["loss"] = torch.stack(list(terms.values())).sum()
    return terms
