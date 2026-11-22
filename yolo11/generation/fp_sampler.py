import torch
import torchvision  # type: ignore

from pydantic import BaseModel

from ultralytics.models.yolo.detect.predict import DetectionPredictor  # type: ignore
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy  # type: ignore
from ultralytics.utils import LOGGER  # type: ignore

from generation.utils import predictions_to_coco
from external.data import LabelBatch
from external.data import merge_to_target_tensor, from_target_tensor


class FpSampler(BaseModel, arbitrary_types_allowed=True):
    enable_box_sampler: bool

    box_sampler_warmup: int
    box_sampler_early_exit: int
    box_sampler_conf: float
    box_sampler_overlap_iou: float
    box_sampler_max_area: float
    box_sampler_min_area: float

    relabel_model: DetectionPredictor

    def sample(
        self, iteration: int, inputs: torch.Tensor, sample_batch: LabelBatch
    ) -> LabelBatch:
        if (
            self.enable_box_sampler is False
            or iteration < self.box_sampler_warmup
            or iteration > self.box_sampler_early_exit
        ):
            return sample_batch
        input_copy = inputs.clone().detach().cpu()
        # get teacher's output
        with torch.no_grad():
            """
            PAST:
            torch.Size([4, 3, 80, 80, 85])
            torch.Size([4, 3, 40, 40, 85])
            torch.Size([4, 3, 20, 20, 85])

            CURRENT:
            torch.Size([4, 144, 80, 80])
            torch.Size([4, 144, 40, 40])
            torch.Size([4, 144, 20, 20])
            """
            preds = self.relabel_model.inference(inputs.clone())[0]
            output = non_max_suppression(
                preds, 0.001, 0.6, classes=None, agnostic=False
            )
        new_targets = predictions_to_coco(output, input_copy)
        new_targets = new_targets[new_targets[:, 2] > self.box_sampler_conf]
        new_targets = torch.index_select(
            new_targets,
            dim=1,
            index=torch.tensor([0, 1, 3, 4, 5, 6]),
        )  # # remove confidence value

        add_targets = torch.zeros((len(new_targets),), dtype=torch.long)

        target = merge_to_target_tensor(sample_batch)
        minus_targets = torch.zeros((len(target),), dtype=torch.long)

        for batch_idx in range(input_copy.shape[0]):
            # todo:check if the float precision is a problem
            _target = target[target[:, 0] == batch_idx]
            _new_targets = new_targets[new_targets[:, 0] == batch_idx]
            # add new target if iou is less than a given point
            if _new_targets.shape[0] > 0 and _target.shape[0] > 0:
                ious = torchvision.ops.box_iou(
                    xywh2xyxy(_new_targets[:, 2:]),
                    xywh2xyxy(_target[:, 2:]),
                )
                assert len(ious.shape) == 2, (
                    f"something is wrong when {_new_targets} {_target} {ious} happens."
                )
                max_ious, _ = torch.max(ious, dim=1)
                _add_targets = (max_ious < self.box_sampler_overlap_iou).long()
                add_targets[new_targets[:, 0] == batch_idx] += _add_targets
                # filter out samples in initial targets
                if _target.shape[0] > 1:
                    initial_ious, _ = torch.max(ious, dim=0)
                    _minus_targets = (
                        initial_ious < self.box_sampler_overlap_iou
                    ).long()
                    if _minus_targets.sum() == _target.shape[0]:
                        max_idx = torch.argmax(initial_ious)
                        _minus_targets[max_idx] = 0
                        print(f"we select idx {max_idx}")
                    minus_targets[target[:, 0] == batch_idx] += _minus_targets

        new_targets = new_targets[add_targets.bool()]
        assert len(new_targets) == add_targets.sum().item()
        areas = new_targets[:, -1] * new_targets[:, -2]
        area_limits = (areas < self.box_sampler_max_area) * (
            areas > self.box_sampler_min_area
        )

        new_targets = new_targets[area_limits.bool()]
        LOGGER.info(
            f"Fp sampling: Minus {minus_targets, minus_targets.sum().item()} old targets to batch for iteration: {iteration} "
        )

        target = target[minus_targets == 0]
        LOGGER.info(
            f"Fp sampling: Adding {len(new_targets), new_targets} new targets to batch for iteration: {iteration} "
        )

        target = torch.cat((target, new_targets), dim=0)
        from_target_tensor(sample_batch, target)

        return sample_batch
