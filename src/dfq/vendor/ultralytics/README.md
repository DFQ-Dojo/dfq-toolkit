# Vendored Ultralytics

Source: [ultralytics 8.3.53](https://pypi.org/project/ultralytics/8.3.53/), AGPL-3.0 (`LICENSE`).

Only what YOLO11 detection needs is here, rewritten to dfq's style, typing and checks:

| File | From | Content |
|---|---|---|
| `nn/modules/conv.py` | `nn/modules/conv.py` | `Conv`, `DWConv`, `Concat` |
| `nn/modules/block.py` | `nn/modules/block.py` | `DFL`, `SPPF`, `Bottleneck`, `C3k`, `C3k2`, `Attention`, `PSABlock`, `C2PSA` |
| `nn/modules/head.py` | `nn/modules/head.py` | `Detect`, without export formats and the end-to-end branch |
| `nn/tasks.py` | `nn/tasks.py` | `DetectionModel`, `parse_model` with an explicit module table |
| `utils/loss.py` | `utils/loss.py` | `v8DetectionLoss`, box and distribution focal losses |
| `utils/tal.py` | `utils/tal.py` | `TaskAlignedAssigner`, anchors, distance and box conversions |
| `utils/metrics.py` | `utils/metrics.py` | Complete IoU |
| `utils/ops.py` | `utils/ops.py` | `non_max_suppression` (single label, class-aware), `xywh2xyxy` |
| `cfg/yolo11.yaml` | `cfg/models/11/yolo11.yaml` | The YOLO11 model definition |

Class and attribute names stay as upstream, because Ultralytics checkpoints pickle these classes
and their state dicts use these names. The rewrite keeps every computation: on 2026-11-22,
YOLO11n's NMS output and training loss matched ultralytics 8.3.53 on the same inputs (see
`docs/research/yolo11-vendoring.md`).
