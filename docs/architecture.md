# Architecture

`dfq` is one Python package under `src/dfq/`. The design and its reasons are in
[`design/unified-package.md`](design/unified-package.md). This page describes what exists now.

## Layers

Modules are listed from stable to fast-changing. A module imports only modules above it.

```text
common/       distributed processes (torchrun), run directories, config files
quant/        fake quantizers (LSQ, LSQ+), quantized Conv2d and Linear, the quantization scope
  detection/    Targets and Detections, the Detector protocol, training modes, output taps,
                the COCO classes
    distill/    distillation losses and objective
    synthesis/  image priors, augmentation, adaptive box sampling, the generation loop
    training/   the quantization-aware training loop, optimizer schedule, weight average
```

`distill/`, `synthesis/` and `training/` depend on the `Detector` protocol, never on a model, and
never on each other. A script connects them: for example, it passes `distill_losses` with a
student and a teacher to `train` as the objective. Each adapter in `models/`
implements the protocol.

## The Detector protocol (`detection/`)

Images are a float batch `[B, 3, H, W]` of RGB values in `[0, 1]`, already resized and padded.
The adapter normalizes them. Boxes are `[x1, y1, x2, y2]` in pixels; labels are contiguous class
indices. `detection/coco_classes.json` is the one list of the 80 COCO classes and their category
ids.

| Member | Meaning |
|---|---|
| `network` | The PyTorch module |
| `num_classes` | The number of object classes |
| `losses(images, targets)` | The detector's training losses; the network is in training mode |
| `predict(images)` | `Detections` per image; the network is in evaluation mode |
| `quant_scope()` | The layers that QAT quantizes |
| `logit_layer()`, `logits(output)` | The module whose output distillation matches, and how to read it |
| `feature_layers(kind)` | The modules whose outputs feature distillation matches |

`generation_mode(network)` puts a network in training mode with its normalization statistics
frozen, which image generation needs. `freeze(network)` stops gradients to its parameters.
`tap(network, names)` records module outputs; its hooks exist only inside its `with` block.

## Quantization (`quant/`)

Quantization-aware training (QAT) trains a model whose layers round their inputs and weights to
low-bit integers in the forward pass. `quant/` does this in three steps:

1. `QuantConfig` is the record that a config file holds: the method (`lsq` or `lsq+`) and the
   weight and activation bit widths. 32 bits leaves a tensor in float.
2. `QuantScope` names the layers to quantize with shell-style patterns over module names, such as
   `include=("model.*",), exclude=("model.0.*",)`. Each detector adapter owns its scope.
3. `quantize(model, config, scope)` replaces each `nn.Conv2d` and `nn.Linear` inside the scope with
   a `QuantConv2d` or `QuantLinear` in place. The quantized layer shares the float layer's weight,
   so quantize a model after loading its float weights.

Each quantized layer has an `input_quantizer` and a `weight_quantizer`. A quantizer sets its step
size from the first tensor it sees, and saves that it has started in the state dict. The exact
forward and backward rules are in the docstring of `quant/functions.py`.

| Method | Offset | Rounding | Gradient factor `g` |
|---|---|---|---|
| `lsq` | none | half to even | recomputed on every call |
| `lsq+` | learned, for activations | half away from zero | fixed on the first batch |

## Distillation (`distill/`)

`distill_losses(student, teacher, images, targets, config)` returns three weighted terms and
their sum, `loss`:

- `detection`: the student's own losses, summed.
- `logit`: KL divergence or DIST between the outputs of `logit_layer`. KL takes a temperature
  `tau` and `softmax_over`: one softmax per output (`each_output`) or one over all outputs joined
  (`all_outputs`).
- `feature`: the mean squared difference between the outputs of `feature_layers(kind)`.

The teacher runs `predict` in evaluation mode without gradients.

## Synthesis (`synthesis/`)

`generate_batch(teacher, targets, size, config, verifier)` optimizes a batch of images for given
boxes and returns the images (pixels in `[0, 1]`), the final boxes, and one `StepRecord` per step.
The module docstring of `synthesis/generate.py` gives the exact objective.

| Config | Meaning |
|---|---|
| `latent` | Mean and standard deviation that map the optimized tensor to pixels |
| `init` | `uniform_pixels`, `normal_pixels` or `normal_latent` |
| `schedule`, `warmup` | `cosine` from the first step, or `warmup_cosine` after `warmup` steps |
| `weights` | Detection loss, TV L1 and L2, batch-norm statistics, first batch norm, attention entropy |
| `augment` | Jitter, flip, brightness, contrast, random erasing |
| `verify_every` | How often the verifier scores the images; the best-scoring images are kept |
| `box_sampling` | The adaptive box sampler and its window; `None` keeps the boxes fixed |

The attention-entropy prior reads the modules that `feature_layers("attention")` names.

## Training (`training/`)

`train(network, objective, batches, config, evaluate, run_dir)` trains a network in place and
returns one `EpochRecord` per epoch. `objective(images, targets)` returns loss terms with their
sum under `"loss"`: `detection_objective(detector)` for plain training, or `distill_losses` bound
to a student and a teacher. `evaluate(network)` returns a score where higher is better, such as
COCO mAP. The loop saves `last.pt` after each epoch and `best.pt` when the score improves.

| Config | Meaning |
|---|---|
| `optimizer` | `sgd`, `adam` or `adamw`; learning rate, momentum, weight decay, `no_decay` patterns |
| `schedule` | Per-epoch factor: `constant`, `linear`, `cosine` (to `final_factor`) or `step` |
| `warmup` | Linear warm-up iterations, start factor, YOLO's bias learning rate and momentum |
| `nominal_batch_size` | Accumulate gradients until this many images, as YOLO does |
| `grad_clip`, `amp`, `ema` | Gradient-norm clipping, mixed precision, the weight average |
| `patience` | Stop after this many epochs without a better score |
| `process_reduction` | Combine the gradients of `torchrun` processes by `mean` or `sum` |

Under `torchrun`, `common/distributed.py` averages gradients after each backward pass; no
`DistributedDataParallel` wrapper hides the network from the taps.
