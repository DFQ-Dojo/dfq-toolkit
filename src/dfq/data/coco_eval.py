"""COCO mean average precision (mAP) of a detector, with pycocotools.

Only the evaluated images count, so a validation subset is scored against its own ground truth.

Detections are mapped back to the original image: boxes lose the letterbox padding, are divided
by the resize scale and clipped, and mask probabilities are cropped to the resized image, resized
to the original size and thresholded at 0.5.
"""

import contextlib
import io
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

from dfq.data.coco import ImageInfo
from dfq.detection import Detections, Detector, Targets
from dfq.detection.coco import coco_classes

STAT_NAMES = ("mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large")


def original_masks(masks: Tensor, info: ImageInfo) -> np.ndarray:
    resized_height = int(info.height * info.scale + 0.5)
    resized_width = int(info.width * info.scale + 0.5)
    cropped = masks[:, :, :resized_height, :resized_width].float()
    restored = F.interpolate(
        cropped, size=(info.height, info.width), mode="bilinear", align_corners=False
    )
    binary = (restored[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return np.asfortranarray(binary.transpose(1, 2, 0))


def image_results(detections: Detections, info: ImageInfo, with_masks: bool) -> list[dict]:
    coco_id = [item.coco_id for item in coco_classes()]
    offset = torch.tensor([info.left, info.top, info.left, info.top])
    boxes = (detections.boxes.detach().cpu() - offset) / info.scale
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, info.width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, info.height)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    scores = detections.scores.detach().cpu().tolist()
    labels = detections.labels.detach().cpu().tolist()
    encoded: list = [None] * len(scores)
    if with_masks and detections.masks is not None and len(scores):
        bitmaps = original_masks(detections.masks, info)
        encoded = mask_utils.encode(bitmaps)
    results = []
    for i, (score, label) in enumerate(zip(scores, labels, strict=True)):
        result = {
            "image_id": info.image_id,
            "category_id": coco_id[label],
            "bbox": [boxes[i, 0].item(), boxes[i, 1].item(), widths[i].item(), heights[i].item()],
            "score": score,
        }
        if encoded[i] is not None:
            rle = encoded[i]
            result["segmentation"] = {"size": rle["size"], "counts": rle["counts"].decode()}
        results.append(result)
    return results


def gathered[T](items: list[T]) -> list[T]:
    if not dist.is_initialized():
        return items
    parts: list[list[T]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(parts, items)
    return [item for part in parts for item in part]


def summarize(
    coco: COCO, results: list[dict], image_ids: list[int], iou_type: str
) -> dict[str, float]:
    """Score `results` against the ground truth of `image_ids` only."""
    if not results:
        return {f"{iou_type}_{name}": 0.0 for name in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        predictions = coco.loadRes(results)
        evaluator = COCOeval(coco, predictions, iou_type)
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        f"{iou_type}_{name}": float(value)
        for name, value in zip(STAT_NAMES, evaluator.stats[:6], strict=True)
    }


def evaluate_coco(
    detector: Detector,
    batches: Iterable[tuple[Tensor, Sequence[Targets], Sequence[ImageInfo]]],
    coco: COCO,
    with_masks: bool,
) -> dict[str, float]:
    """Run `detector` over `batches` and return COCO box (and mask) mAP statistics."""
    device = next(detector.network.parameters()).device
    detector.network.eval()
    results: list[dict] = []
    image_ids: list[int] = []
    with torch.no_grad():
        for images, _, infos in batches:
            detections = detector.predict(images.to(device))
            for found, info in zip(detections, infos, strict=True):
                results.extend(image_results(found, info, with_masks))
                image_ids.append(info.image_id)
    all_results = gathered(results)
    all_ids = gathered(image_ids)
    stats = summarize(coco, all_results, all_ids, "bbox")
    if with_masks:
        stats.update(summarize(coco, all_results, all_ids, "segm"))
    return stats
