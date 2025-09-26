# Tutorial: ResNet

This tutorial provides step-by-step instructions for reproducing the experimental results for the **ResNet** series from our paper.

If you encounter any issues, please open a ticket in our [GitHub Issues](https://github.com/DFQ-Dojo/dfq-toolkit/issues) section.

---

## 🚀 Prerequisites

### Hardware Requirements

- At least **one NVIDIA GPU**
- Minimum **32 GiB of RAM**

### Software Requirements

We use [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.  
To get started, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Once installed, navigate to the `dfq-toolkit/resnet` directory and run:

```bash
uv sync
```

This will install all necessary dependencies and set up a virtual environment.

To activate the environment:

```bash
source .venv/bin/activate
```

> ⚠️ In some regions, `uv sync` may fail due to network restrictions.  
> You can resolve this by configuring a custom [index](https://docs.astral.sh/uv/concepts/indexes/).

---

## 📦 Datasets and Pre-trained Models

Download the pre-trained weights from [this link](https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth).  
Rename the file to `maskrcnn_resnet50_fpn_coco.pth` and place it in the `resnet/` directory.

### Dataset: COCO2017

Expected structure:

```yaml
resnet/
└── coco2017/
    └── images/
        ├── train2017/
        └── val2017/
    └── annotations/
        ├── instances_train2017.json
        ├── instances_val2017.json
        ├── captions_train2017.json
        ├── captions_val2017.json
        ├── person_keypoints_train2017.json
        └── person_keypoints_val2017.json
```

### Dataset: Pascal VOC2012

Expected structure:

```yaml
resnet/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── JPEGImages/
        ├── SegmentationClass/
        ├── SegmentationObject/
        └── ImageSets/
            ├── Action/
            ├── Layout/
            ├── Main/
            │   ├── train.txt
            │   ├── val.txt
            │   └── trainval.txt
            └── Segmentation/
                ├── train.txt
                ├── val.txt
                └── trainval.txt
```

---

## 🧭 Code Structure

| File / Directory     | Description                                       |
|----------------------|---------------------------------------------------|
| `backbone/`          | LSQ ResNet backbones                              |
| `network_files/`     | Mask R-CNN model definitions                      |
| `train_utils/`       | Training utilities: losses, evaluation            |
| `feature_qat.py`     | QAT pipeline with synthetic calibration images    |
| `distill_data.py`    | Real-label calibration image generation           |

---

## 🧪 Stage I: Real-label Based Calibration Set Synthesis

To generate task-specific calibration images using real labels, run:

```bash
python -m distill_data \
    --data-path <path_to_dataset> \
    --device <cuda/cpu> \
    --batch-size 32 \
    --output-dir <output_dir> \
    --calibration_size 500 \
    --iterations 300 \
    --do_clip
```

### 🔧 Key Arguments

| Argument            | Description                                                 |
|---------------------|-------------------------------------------------------------|
| `--data-path`       | Path to the dataset                                         |
| `--device`          | Training device (`cuda:0`, `cpu`, etc.)                    |
| `--batch-size`      | Number of images per batch                                  |
| `--output-dir`      | Directory to save generated calibration images              |
| `--calibration_size`| Number of images to generate                                |
| `--iterations`      | Optimization steps per image                                |
| `--do_clip`         | Clip pixel values to `[0, 1]` after generation              |

---

## 🔧 Stage II: Quantization-Aware Training (QAT) with Task-Specific Distillation

We evaluate two QAT strategies:

1. **QAT using full real dataset** *(baseline)*
2. **QAT using synthetic calibration data** *(ours)*

Run all commands from the `./resnet` directory.

---

### 1. QAT with Real Dataset (Baseline)

```bash
python -m train \
    --device cuda:0 \
    --data-path <path_to_dataset> \
    --output-dir <output_dir> \
    --epochs 100 \
    --batch_size 32 \
    --pretrain maskrcnn_resnet50_fpn_coco.pth \
    --mode quantize \
    --num_bits 8
```

| Argument        | Description                                    |
|------------------|------------------------------------------------|
| `--device`       | Training device (e.g. `cuda:0`)               |
| `--data-path`    | Path to the dataset                           |
| `--output-dir`   | Directory for saving trained weights          |
| `--epochs`       | Maximum number of training epochs             |
| `--batch_size`   | Batch size for training                       |
| `--pretrain`     | Path to the pretrained weights                |
| `--mode`         | Set to `quantize` to enable QAT               |
| `--num_bits`     | Quantization precision (e.g., 8-bit)          |

---

### 2. QAT with Generated Calibration Data (Ours)

```bash
python -m feature_qat \
    --device cuda:0 \
    --data-path <path_to_dataset> \
    --output-dir <output_dir> \
    --epochs 100 \
    --batch_size 32 \
    --pretrain maskrcnn_resnet50_fpn_coco.pth \
    --mode quantize \
    --num_bits 8 \
    --ckpt_path <path_to_generated_images> \
    --kd kdt4+mse \
    --calibration_size 500
```

| Argument              | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `--ckpt_path`          | Path to generated calibration images                                   |
| `--kd`                 | Knowledge distillation method (recommended: `kdt4+mse`)                |
| `--calibration_size`   | Number of calibration images used for training                        |

---

With your setup complete and calibration images ready, you're all set to explore quantization-aware training on ResNet
