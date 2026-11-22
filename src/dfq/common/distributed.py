"""Data-parallel training across processes started by `torchrun`.

Each process holds a full copy of the network. After the backward pass, `average_gradients`
averages the gradients over all processes, so every copy takes the same step. Outside `torchrun`
these functions do nothing.
"""

import os

import torch
import torch.distributed as dist
from torch import nn


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
    return rank() == 0


def start(device: str) -> torch.device:
    """Join the process group when started by `torchrun`; return this process's device."""
    if "LOCAL_RANK" not in os.environ:
        return torch.device(device)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank)


def average_gradients(network: nn.Module) -> None:
    if not dist.is_initialized():
        return
    size = float(dist.get_world_size())
    for parameter in network.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(size)


def broadcast_parameters(network: nn.Module) -> None:
    """Copy the main process's parameters and buffers to all processes."""
    if not dist.is_initialized():
        return
    for tensor in [*network.parameters(), *network.buffers()]:
        dist.broadcast(tensor.data, src=0)
