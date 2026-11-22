"""Record what named modules output during a forward pass.

    with tap(network, ["backbone.stages.0", "rpn_head"]) as outputs:
        network(images)
    outputs["rpn_head"]  # the last output of that module

The hooks exist only inside the `with` block, so a tap never records a forward pass it was not
opened for. A module that runs several times in one pass keeps its last output.
"""

from collections.abc import Generator
from contextlib import contextmanager

from torch import nn


@contextmanager
def tap(network: nn.Module, names: list[str]) -> Generator[dict[str, object]]:
    outputs: dict[str, object] = {}
    handles = []
    for name in names:
        module = network.get_submodule(name)

        def record(_module: nn.Module, _input: object, output: object, name: str = name) -> None:
            outputs[name] = output

        handles.append(module.register_forward_hook(record))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()
