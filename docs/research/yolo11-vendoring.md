# YOLO11 on vendored Ultralytics code

**Question.** Does the rewritten, vendored Ultralytics code compute what ultralytics 8.3.53 does,
and do its checkpoints load without Ultralytics installed?

**Setup.** On 2026-11-22, on CPU. Upstream: ultralytics 8.3.53 with torch 2.5.1 on Python 3.12,
in a separate environment. dfq: torch 2.15 on Python 3.14. Both loaded the official
`yolo11n.pt` (Ultralytics assets v8.3.0); dfq through `scripts/convert_checkpoint.py`. Inputs:
two random 640 x 512 images, and three boxes for the loss.

**Results.**

- NMS output (confidence 0.001, IoU 0.7, 300 boxes): identical (`torch.equal`) for both images.
- Training loss: 36.853287 (dfq) against 36.853279 (upstream); the difference is float noise
  between the two PyTorch versions.
- The converted checkpoint loads with all 499 keys matching.
- Box mAP of YOLO11n on the first 40 COCO val2017 images through `scripts/evaluate.py`: 0.4895.

**What it means.** The vendored code reproduces Ultralytics for inference and the loss. The
checkpoint loader's unpickler refuses classes outside its allow-list
(`tests/models/test_yolo11.py`), so loading a checkpoint cannot run arbitrary code.
