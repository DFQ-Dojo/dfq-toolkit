# The unified quantizers match the old ones

**Question.** Does `dfq.quant` compute the same values and gradients as the five old copies?

**Setup.** On 2026-11-22, a script loaded the old `yolo11/quant/qatops.py` (`LSQconv2d`, mode
`quantize_sym`) and `yolo11/quant/lsqplus.py` (`QuantConv2dPlus`). The old copies in `vit/`,
`resnet/` and `yolov5/` differ from these only in comments, prints and debug plots. The script
built a 3x3 convolution with 8 input and 16 output channels at 4 bits, and ran three SGD steps
(learning rate 0.1) on CPU with batch sizes 2, 3 and 4, against the same convolution quantized by
`dfq.quant` with `lsq` and `lsq+`.

**Result.** Outputs and input gradients were bit-identical (`torch.equal`) at every step, for both
methods. Since the outputs stayed identical after each update, the step-size and offset gradients
matched as well.

**What it means.** Results trained with the old quantizers carry over. Two behaviours changed:

- A quantizer saves whether it has started in the state dict. The old layers re-initialised their
  step sizes on the first forward pass after loading a checkpoint.
- A 32-bit LSQ+ weight quantizer returns the weight. The old one raised a `NameError`.
