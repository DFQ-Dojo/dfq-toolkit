# User manual

This manual covers installing `dfq`, running each script, and reading what it writes. Terms are
defined in [`glossary.md`](glossary.md).

## Install

You need [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU with a CUDA 13.2
driver. From the repository root:

```bash
uv sync --extra cu132   # GPU machine
uv sync --extra cpu     # CPU only, for tests and small runs
```

- `cu132` or `cpu` picks the PyTorch wheels. Pick exactly one.

`uv run` re-syncs the environment without extras before it runs a command, which would swap the
PyTorch build. Run commands with `uv run --no-sync`, or activate the environment once with
`source .venv/bin/activate` and call `python` directly. The examples below use `uv run --no-sync`.

## Data and weights

Scripts read paths from their configs, relative to the repository root. The configs expect:

```text
data/coco/train2017/            COCO 2017 training images
data/coco/val2017/              COCO 2017 validation images
data/coco/annotations/          instances_train2017.json, instances_val2017.json
weights/                        pretrained detectors (below)
```

`tools/coco.sh` downloads COCO 2017. Symbolic links work. Git ignores `data/`, `weights/` and
`outputs/`.

| Family | Weights file | Source |
|---|---|---|
| `resnet50` | `weights/maskrcnn_resnet50_fpn_coco.pth` | [torchvision](https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth) |
| `swin_t` | `weights/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth` | [MMDetection](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth) |
| `swin_s` | `weights/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth` | [MMDetection](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth) |
| `deit_t` | `weights/mask_rcnn_deit_tiny_fpn_3x_coco.pth.tar` | [ViT-Adapter](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask_rcnn_deit_tiny_fpn_3x_coco.pth.tar) |
| `deit_s` | `weights/mask_rcnn_deit_small_fpn_3x_coco.pth.tar` | [ViT-Adapter](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask_rcnn_deit_small_fpn_3x_coco.pth.tar) |
| PSAQ classifier | `weights/swin_tiny_patch4_window7_224.pth` | [Swin release](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) |

| `yolo11n` to `yolo11x` | `weights/yolo11s.pt` (converted, below) | [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| `yolov5n` to `yolov5x` | `weights/yolov5s.pt` (converted, below) | [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) |

MMDetection-format checkpoints load directly; the converter renames their keys on loading.

Ultralytics checkpoints pickle whole modules. Convert each once into a plain state dict:

```bash
uv run --no-sync python scripts/convert_checkpoint.py yolo11s.pt weights/yolo11s.pt
uv run --no-sync python scripts/convert_checkpoint.py --yolov5 yolov5s.pt weights/yolov5s.pt
```

## How the scripts work

Every script takes a TOML config and optional overrides:

```bash
uv run --no-sync python scripts/<script>.py configs/<family>/<stage>.toml [key=value ...]
```

An override is `dotted.key=value`, where the value is written in TOML, so strings need quotes:
`train.epochs=3`, `output="outputs/try"`, `quant.weight_bits=4`. Each run directory receives
`config.json`, the config after overrides.

The method has two stages:

1. **Stage I, generation:** make a label set, then generate calibration images for it.
2. **Stage II, QAT:** train the quantized detector on those images, distilling from the float one.

## `scripts/labels.py`: make a label set

| Config | Source |
|---|---|
| `configs/labels/one_box.toml` | One random box per image, for adaptive box sampling |
| `configs/labels/coco_train.toml` | The boxes of COCO train2017 ("real labels") |

It writes a JSON Lines file of `ImageLabels` records, one per image, with boxes normalized to
`[0, 1]`.

## `scripts/generate.py`: Stage I

```bash
uv run --no-sync python scripts/generate.py configs/resnet50/generate.toml
```

| Config | Generation |
|---|---|
| `configs/resnet50/generate.toml` | Real labels; detection loss, batch-norm prior, TV |
| `configs/swin_t/generate.toml`, `configs/swin_s/generate.toml` | Real labels; detection loss, attention entropy, TV |
| `configs/yolo11s/generate_one_box.toml` | One random box per image at 160 x 160, with adaptive box sampling |
| `configs/yolo11s/generate.toml` | The boxes that `generate_one_box` ended with, at 640 x 640 |
| `configs/yolov5s/generate_one_box.toml`, `generate.toml` | The same two steps for YOLOv5s |

YOLOv5 scales its objectness loss by the image size, so its generation configs set
`model.image_size` to the generated size. The YOLO method runs Stage I in two steps: `labels.py configs/labels/one_box.toml`, then
`generate_one_box.toml`, whose `labels.jsonl` feeds `generate.toml`.

It generates `count` images of `size` from the label set, `batch_size` at a time, and writes a
synthetic set to `output`:

| File | Content |
|---|---|
| `manifest.json` | Image size, batch files, images per batch |
| `batch_NNNNN.pt` | Images (float, `[0, 1]`), boxes in pixels, labels |
| `labels.jsonl` | The final boxes as a label set, to start another generation |
| `history.jsonl` | One `StepRecord` per step: learning rate, weighted loss terms, verifier loss |

A rerun skips the batches that `output` already holds. To use several GPUs, start one process
per GPU with its own `shard_index`, `shard_count`, `output` and `device`:

```bash
for gpu in 0 1 2 3; do
  uv run --no-sync python scripts/generate.py configs/resnet50/generate.toml \
    shard_index=$gpu shard_count=4 device=\"cuda:$gpu\" output=\"outputs/resnet50/generated/$gpu\" &
done
```

Stage II then lists all parts in `data.directories`.

## `scripts/psaq.py`: the PSAQ-ViT baseline

```bash
uv run --no-sync python scripts/psaq.py configs/baselines/psaq.toml
uv run --no-sync python scripts/qat.py configs/swin_t/qat_ours.toml \
  data.directories=[\"outputs/baselines/psaq\"] output=\"outputs/swin_t/qat_psaq\"
```

It writes a synthetic set of 640 x 640 images with one dummy box each, which Stage II trains on
like generated images. `load_final_norm = true` loads the classifier's final norm, which the
paper's baseline lacked ([details](research/transformer-adapter-checks.md)).

## `scripts/qat.py`: Stage II

```bash
uv run --no-sync python scripts/qat.py configs/resnet50/qat_ours.toml
uv run --no-sync torchrun --nproc-per-node 8 scripts/qat.py configs/resnet50/qat_ours.toml
```

| Config | Run |
|---|---|
| `configs/resnet50/qat_ours.toml` | 8-bit LSQ on generated images, with distillation (the method) |
| `configs/resnet50/qat_real.toml` | 8-bit LSQ on COCO train2017, without distillation (baseline) |
| `configs/swin_t/qat_ours.toml`, `configs/swin_s/qat_ours.toml` | The method on Swin-T and Swin-S |
| `configs/swin_*/qat_real_2k.toml` | Baseline: the first 2048 COCO train2017 images |
| `configs/swin_*/qat_real_full.toml`, `configs/deit_t/qat_real_full.toml` | Baseline: all of COCO train2017 |

| `configs/yolo11s/qat_ours.toml` | The method on YOLO11s |
| `configs/yolo11s/qat_real_2k.toml`, `qat_real_full.toml` | Baselines: 2% and all of COCO train2017 |
| `configs/yolov5s/qat_ours.toml`, `qat_real_full.toml` | The method and the full-data baseline on YOLOv5s |

The YOLO real-data baselines letterbox and flip; the old Ultralytics trainer also used mosaic,
perspective and colour augmentation, which dfq does not implement yet.

For DeiT-S or another variant, override the family and weights:
`model.family="deit_s" model.weights="weights/mask_rcnn_deit_small_fpn_3x_coco.pth.tar"`.

Set `quant.method` to `lsq` or `lsq+`, and `quant.weight_bits` and `quant.activation_bits`. After
each epoch the run scores COCO box mAP on `validation` and writes to `output`:

| File | Content |
|---|---|
| `last.pt`, `best.pt` | State dicts of the last and the best-scoring epoch |
| `epochs.jsonl` | One `EpochRecord` per epoch: learning rate, mean loss terms, score |
| `result.json` | The best box mAP and the number of epochs |

`validation.limit=500` scores only the first 500 validation images, which is faster.

## `scripts/evaluate.py`: COCO mAP

```bash
uv run --no-sync python scripts/evaluate.py configs/resnet50/evaluate.toml
uv run --no-sync python scripts/evaluate.py configs/resnet50/evaluate.toml \
  'quant={method="lsq",weight_bits=8,activation_bits=8}' checkpoint=\"outputs/resnet50/qat_ours/best.pt\" \
  --output outputs/resnet50/qat_ours/val.json
```

It prints box and mask mAP, mAP50, mAP75 and mAP by object size, and with `--output` writes them
as JSON.

## Checks

```bash
uv run --no-sync ruff check && uv run --no-sync ruff format --check && uv run --no-sync mypy \
  && uv run --no-sync python -m unittest
```

The unit tests run on CPU in under a minute and need no data or weights.
