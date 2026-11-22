# Vendored YOLOv5

Source: [Ultralytics YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0), AGPL-3.0
(`LICENSE`), as copied into the old `yolov5/` directory of this repository.

Only what YOLOv5 detection needs is here, rewritten to dfq's style, typing and checks:

| File | From | Content |
|---|---|---|
| `common.py` | `models/common.py` | `Bottleneck`, `C3`; `Conv`, `Concat`, `SPPF` are the vendored Ultralytics ones |
| `yolo.py` | `models/yolo.py` | `Detect`, `DetectionModel` (pickled as `Model`), `parse_model` |
| `loss.py` | `utils/loss.py`, `utils/metrics.py` | `ComputeLoss` without focal loss, label smoothing and auto-balance; centre-box CIoU |
| `ops.py` | `utils/general.py` | `non_max_suppression` (single label, class-aware) |
| `yolov5.yaml` | `models/yolov5{n,s,m,l,x}.yaml` | One definition with each size's depth and width multiples |

Class and attribute names stay as upstream, because YOLOv5 checkpoints pickle these classes. On
2026-11-22, YOLOv5s's NMS output and loss terms matched the old code on the same inputs (see
`docs/research/yolov5-vendoring.md`).
