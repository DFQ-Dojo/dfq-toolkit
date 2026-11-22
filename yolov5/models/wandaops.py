import torch
import torch.nn as nn
import torch.nn.functional as F


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp):
        # inp shape in Conv2d: [batch_size, in_channel, height, width]
        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        C_in = inp.shape[1]
        if len(inp.shape) == 4:
            inp = (
                inp.permute(1, 0, 2, 3).contiguous().view(C_in, -1)
            )  # -> shape: [in_channel, batch_size * height * width]

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        assert self.scaler_row.shape == torch.Size([C_in]), f"scaler shape is {self.scaler_row.shape}"


class WandaConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        prune_n=0,
        prune_m=0,
        sparsity_ratio=0.1,
    ):
        super(WandaConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.prune = False
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.sparsity_ratio = sparsity_ratio
        self.prune_mode = "output"

    def start_prune(self, sparsity=None, prune_mode=None):
        self.prune = True
        self.wrapped_gpt = WrappedGPT(self)
        if sparsity is not None:
            self.sparsity_ratio = sparsity
            # print(f'sparsity is {self.sparsity_ratio}')
        if prune_mode is not None:
            self.prune_mode = prune_mode

    def end_calibration(self):
        scaler = self.wrapped_gpt.scaler_row[None, :, None, None].expand(self.weight.shape)
        W_metric = torch.abs(self.weight.data) * torch.sqrt(scaler)  # [out_channel, in_channel, kernel, kernel]
        if self.prune_n != 0:
            # todo: structured n:m sparsity
            W_metric = W_metric.reshape(W_metric.shape[0], -1)  # [out_channel, in_channel * kernel * kernel]
            W_mask = torch.zeros_like(W_metric) == 1
            for ii in range(W_metric.shape[1]):
                if ii % self.prune_m == 0:
                    tmp = W_metric[:, ii : (ii + self.prune_m)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1],
                        True,
                    )
            W_mask = W_mask.view(self.weight.shape)

        else:
            # W_metric = W_metric.permute(1,0,2,3).contiguous().view(self.weight.shape[1], -1) # [in_channel, out * kernel * kernel]
            if self.prune_mode == "output":
                W_metric = W_metric.reshape(W_metric.shape[0], -1)  # [out_channel, in_channel * kernel * kernel]

                W_mask = torch.zeros_like(W_metric) == 1
                # sort as in channel dimension
                sort_res = torch.sort(W_metric, dim=1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)  # [in_channel, out * kernel * kernel]
                # W_mask = W_mask.view(torch.transpose(self.weight, 0, 1).shape).permute(1,0,2,3).contiguous() # [out_channel, in_channel, kernel, kernel]
                W_mask = W_mask.view(self.weight.shape)  # [out_channel, in_channel, kernel, kernel]
            elif self.prune_mode == "layer":
                W_metric = W_metric.reshape(-1)
                W_mask = torch.zeros_like(W_metric) == 1
                sort_res = torch.sort(W_metric, stable=True)
                indices = sort_res[1][: int(W_metric.shape[0] * self.sparsity_ratio)]
                W_mask.scatter_(0, indices, True)
                W_mask = W_mask.view(self.weight.shape)

        self.weight.data[W_mask] = 0

    def check_sparsity(self):
        sub_count = 0
        sub_params = 0

        W = self.weight.data
        sub_count += (W == 0).sum().item()
        sub_params += W.numel()

        print(f"sparsity {float(sub_count) / sub_params:.6f}")

    def forward(self, input):
        if self.prune:
            self.wrapped_gpt.add_batch(input.data)
        output = F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output
