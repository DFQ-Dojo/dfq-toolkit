"""Training-mode settings that the pipelines need beyond `train()` and `eval()`."""

from torch import nn
from torch.nn.modules.batchnorm import _NormBase


def generation_mode(network: nn.Module) -> None:
    """Put `network` in training mode with its normalization statistics frozen.

    Image generation needs the detector's training losses, but must not change the batch-norm
    running statistics, which its prior compares the images against.
    """
    network.train()
    for module in network.modules():
        if isinstance(module, _NormBase):
            module.eval()


def freeze(network: nn.Module) -> None:
    """Stop gradients to every parameter of `network`, such as a teacher or a generator's model."""
    for parameter in network.parameters():
        parameter.requires_grad_(False)
