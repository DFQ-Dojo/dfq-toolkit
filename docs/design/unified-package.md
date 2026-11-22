# One package for all four detectors

Status: Decided, 2026-11-22.

## Problem

The repository holds four separate projects (`resnet/`, `vit/`, `yolo11/`, `yolov5/`). Each has its own
Python version (3.8 to 3.12), its own PyTorch (1.9 to 2.5) and its own copy of the method:

- The quantizers (LSQ and LSQ+) exist in five copies that have drifted apart.
- The distillation losses (KL divergence, DIST, MSE) exist in four copies.
- `vit/` depends on MMDetection 2 and mmcv-full 1.4. Neither is maintained, and neither installs on
  Python 3.14.
- `yolov5/` vendors the whole upstream repository. It encodes the bit width in 20 model YAML files, one
  per size and bit width.

## Decision

One installable package, `dfq`, under `src/dfq/`, with one `pyproject.toml`, managed by uv, on Python
3.14.

- **Decision:** replace MMDetection with torchvision's Mask R-CNN. The Swin and DeiT backbones are
  ported to plain PyTorch, so that their checkpoint keys stay the same, and a converter maps the
  MMDetection head weights to torchvision. **Rejected:** MMDetection 3, which is also dormant and
  needs mmcv, which has no Python 3.14 build.
- **Decision:** the project owns the training and generation loops. Ultralytics, torchvision and timm
  provide models, losses and layers, not the workflow. **Rejected:** keeping the four trainers (the
  Ultralytics trainer subclass, the MMDetection runner, two hand-written loops). Each would need its
  own copy of the distillation hooks.
- **Decision:** quantization is one policy applied to a built model by swapping modules. **Rejected:**
  model definitions that carry bit widths, such as the per-bit YAML files.
- **Decision:** depend only on stable, widely used packages: PyTorch, torchvision, NumPy, Pillow,
  PyYAML, Pydantic and pycocotools. **Rejected:** depending on
  Ultralytics and timm, whose future and interfaces we cannot rely on.
- **Decision:** vendor the parts of Ultralytics (8.3.53, the version the YOLO11 results used),
  YOLOv5 and timm that the pipeline imports, under `dfq/vendor/`, each with its licence and source
  version. Delete the rest, such as export, benchmarks, hub, segmentation and classification.
  Vendored code is our code: it follows the same style, typing and checks as the rest of `dfq`.
- **Decision:** checkpoints become plain state dicts. Ultralytics and YOLOv5 save pickled modules,
  which name their import paths (`ultralytics.nn.tasks`, `models.yolo`). A remapping unpickler loads
  them into the vendored classes, and `scripts/convert_checkpoint.py` saves them again as a state
  dict and a model YAML. **Rejected:** keeping pickled modules, which break whenever a class moves.
- **Decision:** keep the baselines: PSAQ-ViT, the DeiT configs, and the real-data (2k and full COCO)
  quantization-aware training runs.

## Layers

Modules are listed from stable to fast-changing. A module imports only modules above it.

```text
common/       config files, run directories, seeding, distributed helpers
  quant/      fake quantizers, quantized layers, the quantization policy
  detection/  the Detector protocol, target and prediction records, COCO evaluation
    data/       COCO datasets, label files, synthetic image sets
    distill/    KD losses, feature taps, the distillation objective
    synthesis/  image priors, box sampling, the generation loop
    training/   the quantization-aware training loop
  models/     one adapter per detector family; each implements Detector
baselines/    PSAQ-ViT
vendor/       frozen third-party code: ultralytics, yolov5, timm; only models/ and baselines/ import it
```

- `models/` never imports `data/`, `distill/`, `synthesis/` or `training/`, and they never import
  `models/`. Only scripts connect them. This is the dependency inversion: the pipelines depend on the
  `Detector` protocol, and the adapters implement it.
- `scripts/` are thin. They read a TOML config, build the objects, call `src/` and write outputs.
- Persisted data (configs, label files, synthetic set manifests, results) are Pydantic records that
  round-trip through JSON or TOML. Runtime collaborators (models, losses, priors) are protocols or
  ordinary classes.

## The Detector protocol

Every pipeline needs four things from a detector. The adapter provides them, so the pipelines never
know which family they run.

| Member | Used by | Meaning |
|---|---|---|
| `losses(images, targets)` | synthesis, training | The detector's own training losses, as a dict of scalars |
| `predict(images)` | evaluation | Boxes, scores and labels per image |
| `logits()` | distill | The raw head outputs of the last forward pass, for the KL term |
| `quant_rules()` | quant | Which modules to quantize and which to keep at full precision |

## Migration

All six steps are done (2026-11-22). Each detector family moved in its own commits: adapter, configs, scripts and tests first, then the
old directory is deleted in the commit that replaces its last entry point. Each commit stays under
about 1000 changed lines, not counting moved files.

1. Package skeleton, `pyproject.toml`, checks.
2. `quant/`, `distill/`, `detection/`, `data/`, `synthesis/`, `training/`, with CPU unit tests.
3. ResNet Mask R-CNN on torchvision; delete `resnet/`.
4. Swin and DeiT Mask R-CNN on torchvision, the checkpoint converter and PSAQ; delete `vit/`.
5. YOLO11 on Ultralytics; delete `yolo11/`.
6. Trimmed YOLOv5; delete `yolov5/`.

## Limitations

- The torchvision Mask R-CNN differs from MMDetection's in small ways: L1 against smooth L1 box
  losses, rounded anchor corners, and low-quality matches in the box head. Pooling uses
  MMDetection's aligned RoIAlign. Converted Swin and DeiT checkpoints therefore give
  slightly different COCO mAP. The released weights must be evaluated again on a GPU, and the new
  numbers recorded in `docs/research/`.
- The unified training loop replaces the Ultralytics and MMDetection trainers. It reproduces their
  warm-up, schedule, EMA and mixed precision from config, but the paper's numbers must be reproduced
  again before they are claimed for the new code.
- The YOLO real-data baselines (2% and full COCO) letterbox and flip only. The old Ultralytics
  trainer also used mosaic, perspective and colour augmentation; until those are vendored, those
  baselines are not comparable with the paper's.
