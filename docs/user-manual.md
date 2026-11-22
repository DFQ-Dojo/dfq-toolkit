# User manual

This manual covers installing `dfq`, running each script, and reading what it writes. Terms are
defined in [`glossary.md`](glossary.md).

## Install

You need [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU with a CUDA 13.2
driver. From the repository root:

```bash
uv sync --extra cu132 --extra vit --extra yolo   # GPU machine
uv sync --extra cpu --extra vit --extra yolo     # CPU only, for tests and small runs
```

- `cu132` or `cpu` picks the PyTorch wheels. Pick exactly one.
- `vit` adds what the Swin and DeiT models need; `yolo` adds what the YOLO models need.

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

## `scripts/qat.py`: Stage II

```bash
uv run --no-sync python scripts/qat.py configs/resnet50/qat_ours.toml
uv run --no-sync torchrun --nproc-per-node 8 scripts/qat.py configs/resnet50/qat_ours.toml
```

| Config | Run |
|---|---|
| `configs/resnet50/qat_ours.toml` | 8-bit LSQ on generated images, with distillation (the method) |
| `configs/resnet50/qat_real.toml` | 8-bit LSQ on COCO train2017, without distillation (baseline) |

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
