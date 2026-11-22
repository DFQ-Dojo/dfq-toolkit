# Glossary

| Term | Meaning |
|---|---|
| **Adapter** | A class in `src/dfq/models/` that makes one detector family satisfy the `Detector` protocol |
| **AMP** | Automatic mixed precision: running parts of training in 16-bit floats |
| **BN prior** | A generation loss that compares each batch-norm layer's input statistics with its running statistics |
| **Box sampling** | During generation, replacing boxes with the verifier's confident, new detections |
| **Calibration set** | The images that quantization-aware training uses; here, generated images |
| **COCO** | Common Objects in Context, the detection dataset with 80 object classes |
| **DIST** | A distillation loss that matches inter- and intra-class relations ([paper](https://arxiv.org/abs/2205.10536)) |
| **Distillation (KD)** | Training a student to match a teacher's outputs as well as the labels |
| **EMA** | Exponential moving average of the weights, which YOLO training evaluates |
| **Family** | A detector architecture with its weights, such as `resnet50` |
| **KL divergence** | A distance between two probability distributions, used as the logit distillation loss |
| **Label set** | A JSON Lines file of boxes, normalized to `[0, 1]`, that generation starts from |
| **LSQ, LSQ+** | Learned step-size quantization, without and with a learned offset |
| **mAP** | Mean average precision, COCO's detection score; `mAP50` counts a box as correct at IoU 0.5 |
| **Prior** | A generation loss that keeps images realistic, such as total variation |
| **QAT** | Quantization-aware training: training a model whose layers simulate low-bit arithmetic |
| **Recipe** | A function in `src/dfq/recipes/` that runs one stage from one config record |
| **RPN** | Region proposal network: the Mask R-CNN head whose outputs the logit distillation matches |
| **Scope** | The layers a model quantizes, named by patterns over module names |
| **Synthetic set** | A directory of generated images, their boxes and a manifest |
| **Teacher, student** | The full-precision detector, and its quantized copy that learns from it |
| **TV** | Total variation: the summed difference between neighbouring pixels |
| **Verifier** | The detector that scores generated images to keep the best and to sample boxes |
