# Architecture

`dfq` is one Python package under `src/dfq/`. The design and its reasons are in
[`design/unified-package.md`](design/unified-package.md). This page describes what exists now.

## Layers

Modules are listed from stable to fast-changing. A module imports only modules above it.

```text
quant/   fake quantizers (LSQ, LSQ+), quantized Conv2d and Linear, the quantization scope
```

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
