# ResNet-50 Mask R-CNN on the new package

**Question.** Does the torchvision adapter predict what stock torchvision predicts, and does the
pipeline run end to end?

**Setup.** On 2026-11-22, on CPU, with torchvision's `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` and
the first 40 images of COCO val2017 (by image id). Images were resized to 800 x 1333 and padded
with the pixel mean.

**Results.**

- On 5 images, the adapter and stock torchvision with resizing turned off returned the same
  number of detections, with identical boxes and scores (largest difference 0.0).
- On the 40 images, the float model scored box mAP 0.4625 and mask mAP 0.4045. These 40 images are
  too few to compare with the published full-val numbers (37.9 box, 34.6 mask).
- An end-to-end run (labels from the 40 images, 4 images generated at 256 x 256 in 3 steps, 1 QAT
  epoch with distillation, evaluation on 4 images) completed. On those 4 images the float model
  scored box mAP 0.488 and the 8-bit LSQ student 0.440.

**What it means.** The adapter reproduces torchvision. The end-to-end run checks that the stages
connect; it says nothing about the method's accuracy, which needs the full runs on a GPU.

The run also exposed that COCO evaluation scored a subset against the whole annotation file. It
now scores only the evaluated images.
