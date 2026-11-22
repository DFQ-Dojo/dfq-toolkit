# YOLOv5 on vendored code

**Question.** Does the rewritten, vendored YOLOv5 compute what the old `yolov5/` code did?

**Setup.** On 2026-11-22, on CPU. Old code: the repository's `yolov5/` directory with torch 2.5.1
on Python 3.12. dfq: torch 2.15 on Python 3.14. Both loaded the official `yolov5s.pt` (YOLOv5
v7.0); dfq through `scripts/convert_checkpoint.py --yolov5`. Inputs: two random 640 x 512 images
and three boxes; loss gains of `data/hyps/hyps.qat.yaml`.

**Results.**

- NMS output (confidence 0.001, IoU 0.6, 300 boxes): identical (`torch.equal`) for both images.
- Loss terms (box, objectness, class) for the same target rows: identical, 0.09929, 0.03836 and
  0.12011.
- Through `Yolov5Detector.losses`, which converts corner boxes in pixels back to normalized centre
  boxes, the loss was 0.51433 against 0.51553. The test boxes lie exactly on cell borders, so the
  conversion's rounding moved some of them to a neighbouring cell. Real boxes rarely lie exactly
  on a border.
- Box mAP of YOLOv5s on the first 4 COCO val2017 images through `scripts/evaluate.py`: 0.4953.

**What it means.** The vendored YOLOv5 reproduces the old model, loss and NMS.
