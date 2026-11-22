# Swin, DeiT and PSAQ without MMDetection

**Question.** Do the plain-PyTorch Swin and DeiT backbones, the torchvision heads and the
checkpoint converter reproduce the released detectors, and does the PSAQ-ViT classifier load?

**Setup.** On 2026-11-22, on CPU, with the first 40 images of COCO val2017 (by image id), resized
to 800 x 1333 and padded with the pixel mean. Checkpoints:

- Swin-T: `mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth`
  (MMDetection);
- DeiT-T: `mask_rcnn_deit_tiny_fpn_3x_coco.pth.tar` (ViT-Adapter v0.3.1);
- PSAQ classifier: `swin_tiny_patch4_window7_224.pth` (official Swin release).

**Results.**

| Detector | Box mAP | Mask mAP | Published, full val (box / mask) |
|---|---|---|---|
| ResNet-50 (torchvision) | 0.4625 | 0.4045 | 37.9 / 34.6 |
| DeiT-T | 0.4686 | 0.4309 | |
| Swin-T | 0.5945 | 0.5299 | 46.0 / 41.7 |

The 40 images are too few to compare with published numbers. But a wrong key mapping or class
order would score near zero, and Swin-T beats ResNet-50 by a similar margin as published.

The PSAQ classifier ranks the right ImageNet class first on two COCO images (brown bear, street
sign). With the old code's identity `norm3`, the top-1 probability is 0.066 and 0.065; with the
checkpoint's final norm loaded, 0.978 and 0.915.

A unit test checks that the reordered patch-merging weights reproduce the official Swin patch
merging (`tests/baselines/test_psaq.py`).

**Findings about the old code.**

- **Quantized DeiT lost its windows.** The old `quant_vit.py` rebuilt every block's attention as
  global attention, so the quantized DeiT students attended globally in all 12 blocks, while their
  teachers used 14 x 14 windows in 8 of them. `dfq.quant` keeps each block's attention.
- **PSAQ ran without the final norm.** The checkpoint's `norm` matched no MMDetection key, so the
  PSAQ classifier used an identity `norm3`. `configs/baselines/psaq.toml` keeps this by default;
  `load_final_norm = true` runs the corrected baseline.

**What it means.** The ViT track runs without MMDetection. Its QAT numbers must be measured again
on a GPU: the heads differ in small ways (see `design/unified-package.md`), and the DeiT students
now keep windowed attention.
