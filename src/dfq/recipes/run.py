"""The four stages: make labels, generate images, train with quantization, evaluate."""

import functools
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from dfq.baselines.psaq import dummy_targets, generate_psaq_batch, load_classifier
from dfq.common import distributed
from dfq.common.runs import seed_everything, start_run, write_json, write_records
from dfq.data.coco_eval import evaluate_coco
from dfq.data.labels import coco_labels, one_box_labels, read_label_set, to_targets, write_label_set
from dfq.data.synthetic import append_batch, read_manifest, start_set
from dfq.detection import Detector, targets_to
from dfq.distill import distill_losses
from dfq.recipes.configs import (
    EvaluateConfig,
    GenerateConfig,
    LabelsConfig,
    PsaqConfig,
    QatConfig,
)
from dfq.recipes.data import train_batches, validation_batches
from dfq.recipes.families import float_detector, is_yolo, quantized_detector
from dfq.synthesis import generate_batch
from dfq.training import detection_objective, train


def make_labels(config: LabelsConfig) -> int:
    if config.source == "one_box":
        items = one_box_labels(config.count, config.num_classes, config.seed)
    else:
        assert config.annotations is not None, "coco labels need an annotation file"
        items = coco_labels(config.annotations)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    write_label_set(config.output, items)
    return len(items)


def generate(config: GenerateConfig) -> int:
    """Generate the images of this shard; return how many the output holds."""
    device = torch.device(config.device)
    seed_everything(config.seed + config.shard_index)
    start_run(config.output, config)
    height, width = config.size
    if not (config.output / "manifest.json").exists():
        start_set(config.output, height, width)
    labels = read_label_set(config.labels)[: config.count]
    mine = labels[config.shard_index :: config.shard_count]
    teacher = float_detector(config.model, device)
    verifier = teacher if config.verify else None
    done = sum(read_manifest(config.output).counts)
    for start in range(done, len(mine), config.batch_size):
        chunk = mine[start : start + config.batch_size]
        targets = [targets_to(to_targets(item, height, width), device) for item in chunk]
        batch = generate_batch(teacher, targets, config.size, config.generation, verifier)
        append_batch(config.output, batch.images, batch.targets)
        write_records(config.output / "history.jsonl", batch.history)
    return sum(read_manifest(config.output).counts)


def psaq(config: PsaqConfig) -> int:
    """Generate the PSAQ-ViT baseline's images; return how many the output holds."""
    device = torch.device(config.device)
    seed_everything(config.seed)
    start_run(config.output, config)
    if not (config.output / "manifest.json").exists():
        start_set(config.output, 640, 640)
    classifier = load_classifier(config.weights, config.load_final_norm).to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    done = sum(read_manifest(config.output).counts)
    for start in range(done, config.count, config.batch_size):
        size = min(config.batch_size, config.count - start)
        images = generate_psaq_batch(classifier, size, config.steps, device)
        append_batch(config.output, images, dummy_targets(size))
    return sum(read_manifest(config.output).counts)


def with_network(detector: Detector, network: nn.Module) -> Detector:
    """The same adapter around another copy of its network, such as the weight average."""
    copied: Detector = replace(detector, network=network)  # type: ignore[type-var]
    return copied


def qat(config: QatConfig) -> dict[str, float]:
    device = distributed.start(config.device)
    seed_everything(config.seed + distributed.rank())
    if distributed.is_main():
        start_run(config.output, config)
    student = quantized_detector(config.model, config.quant, None, device)
    distributed.broadcast_parameters(student.network)
    objective = detection_objective(student)
    if config.distill is not None:
        teacher = float_detector(config.model, device)
        objective = functools.partial(distill_losses, student, teacher, config=config.distill)
    yolo = is_yolo(config.model.family)
    batches = train_batches(config.data, config.resize, yolo, config.batch_size, config.workers)
    validation, samples = validation_batches(config.validation, yolo, 1, config.workers)

    def score(network: nn.Module) -> float:
        stats = evaluate_coco(
            with_network(student, network), validation, samples.coco, with_masks=False
        )
        return stats["bbox_mAP"]

    records = train(student.network, objective, batches, config.train, score, config.output)
    best = max((r.score for r in records if r.score is not None), default=0.0)
    if distributed.is_main():
        write_records(config.output / "epochs.jsonl", records)
        write_json(config.output / "result.json", {"best_bbox_mAP": best, "epochs": len(records)})
    return {"best_bbox_mAP": best}


def evaluate(config: EvaluateConfig, output: Path | None) -> dict[str, float]:
    device = distributed.start(config.device)
    if config.quant is None:
        detector = float_detector(config.model, device)
    else:
        detector = quantized_detector(config.model, config.quant, config.checkpoint, device)
    yolo = is_yolo(config.model.family)
    batches, samples = validation_batches(
        config.validation, yolo, config.batch_size, config.workers
    )
    stats = evaluate_coco(detector, batches, samples.coco, with_masks=not yolo)
    if output is not None and distributed.is_main():
        write_json(output, stats)
    return stats
