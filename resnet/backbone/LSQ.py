import torch

# import actnn.cpp_extension.backward_func as ext_backward_func
import torch.nn.functional as F
import torch.nn as nn
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os


# Symmetric layer-wise LsqQuantizer, mainly for weight
class SymLsqQuantizer(torch.autograd.Function):
    """
    Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, scale, num_bits):
        """
        :param input: input to be quantized
        :param scale: the step size
        :param num_bits: quantization bits
        :return: quantized output
        """
        try:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

            assert scale.min() > 0, (
                "step size = {:.6f} becomes non-positive".format(scale)
            )
            times = 1.0 / math.sqrt(input.numel() * Qp)
            ctx.save_for_backward(input, scale)
            ctx.other = times, Qn, Qp

            q_w = (input / scale).round().clamp(Qn, Qp)
            w_q = q_w * scale

            return w_q
        except ZeroDivisionError:
            print("zero devide happens")
            import IPython

            IPython.embed()

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale = ctx.saved_tensors
        times, Qn, Qp = ctx.other

        q_w = input_ / scale

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)

        grad_scale = (
            (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w + q_w.round())
                )
                * grad_output
                * times
            )
            .sum()
            .unsqueeze(dim=0)
        )

        grad_input = indicate_middle * grad_output

        return grad_input, grad_scale, None


# Asymmetric layer-wise LsqQuantizer, mainly for activation
class AsymLsqQuantizer(torch.autograd.Function):
    """
    Asymetric LSQ quantization. Modified from LSQ.
    """

    @staticmethod
    def forward(ctx, input, scale, num_bits):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        Qn = 0
        Qp = 2 ** (num_bits) - 1
        # asymmetric: make sure input \in [0, +\inf], remember to add it back
        min_val = input.min().item()
        input_ = input - min_val
        assert scale.min() > 0, (
            "step size = {:.6f} becomes non-positive".format(scale)
        )
        times = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, scale)
        ctx.other = times, Qn, Qp

        q_w = (input / scale).round().clamp(Qn, Qp)
        w_q = q_w * scale

        w_q = w_q + min_val

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale = ctx.saved_tensors
        times, Qn, Qp = ctx.other

        q_w = input_ / scale

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)

        grad_scale = (
            (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w + q_w.round())
                )
                * grad_output
                * times
            )
            .sum()
            .unsqueeze(dim=0)
        )

        grad_input = indicate_middle * grad_output

        return grad_input, grad_scale, None


"""
    mode : - exact : use F.conv2d
           - quantize_sym : use sym LSQ for both activation and weight
           - quantize_asym : use asym LSQ for activation , sym LSQ for weight
"""


class LSQconv2d(nn.Conv2d):
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
        a_bits=4,
        w_bits=4,
        mode="lsq_sym",
        name="",
    ):
        super(LSQconv2d, self).__init__(
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
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.mode = mode
        self.active_track, self.weight_track, self.iter_track = [], [], []
        self.scale_input = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scale_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        assert self.scale_input.requires_grad, "step size needs grad!"
        assert self.scale_weight.requires_grad, "step size needs grad!"
        self.start_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        # self.quantize = False
        self.quantize = True
        self.first_pass = False
        self.name = name
        self.clip = 0.0
        if "clip" in self.mode:
            self.clip = float(self.mode.split("clip")[1])

    # def start_quantize(self):
    #     if self.mode == 'exact':
    #         return
    #     self.quantize = True
    #     print('start quantize')

    def forward(self, input):
        if self.quantize and self.mode.startswith("lsq_"):
            if not self.first_pass:
                Qsym_w = 2 ** (self.w_bits - 1) - 1
                Qsym_a = 2 ** (self.a_bits - 1) - 1
                Qasym_a = 2**self.a_bits - 1
                self.scale_weight.data.copy_(
                    2 * self.weight.abs().mean() / math.sqrt(Qsym_w) + 1e-10
                )
                if self.mode.startswith("lsq_sym"):
                    self.scale_input.data.copy_(
                        2 * input.abs().mean() / math.sqrt(Qsym_a) + 1e-10
                    )
                elif self.mode.startswith("lsq_asym"):
                    self.scale_input.data.copy_(
                        2 * input.abs().mean() / math.sqrt(Qasym_a) + 1e-10
                    )
                else:
                    raise NotImplementedError
                print(
                    f"Actually Using LSQconv2d! Init scale_input = {self.scale_input.data.item()} Init scale_weight = {self.scale_weight.data.item()}"
                )
                print(f"a_bit={self.a_bits}, w_bit={self.w_bits}")
                self.first_pass = True
            else:
                self.scale_input.data = self.scale_input.data.abs()
                self.scale_weight.data = self.scale_weight.data.abs()

            # if self.first_pass:
            #     self.active_track.append(self.scale_input.cpu().detach().numpy())
            #     self.weight_track.append(self.scale_weight.cpu().detach().numpy())
            #     self.iter_track.append(len(self.iter_track))
            # if len(self.iter_track) % 20000 == 5000:
            #     self.draw_clip_value()

        if input.shape[0] == 0:
            print("Oh my god!")
            import IPython

            IPython.embed()
        if self.mode == "exact":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.mode.startswith("lsq_sym"):
            return F.conv2d(
                SymLsqQuantizer.apply(input, self.scale_input, self.a_bits),
                SymLsqQuantizer.apply(
                    self.weight, self.scale_weight, self.w_bits
                ),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.mode.startswith("lsq_asym"):
            return F.conv2d(
                AsymLsqQuantizer.apply(input, self.scale_input, self.a_bits),
                SymLsqQuantizer.apply(
                    self.weight, self.scale_weight, self.w_bits
                ),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            raise NotImplementedError

    # Todo: how to only draw one pic when using parallel training?
    def draw_clip_value(self):

        plt.figure()
        plt.title("{}\n{}".format(self.active_track[0], self.active_track[-1]))
        plt.plot(self.iter_track, self.active_track)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists(
            "plt/step_size_conv/input/{}".format(self.start_time)
        ):
            print(
                "Directory {} created".format(
                    "plt/step_size_conv/input/{}".format(self.start_time)
                )
            )

        os.makedirs(
            "plt/step_size_conv/input/{}".format(self.start_time), exist_ok=True
        )
        plt.savefig(
            "plt/step_size_conv/input/{}/{}.png".format(
                self.start_time, len(self.iter_track)
            )
        )
        plt.close()

        plt.figure()
        plt.title("{}\n{}".format(self.weight_track[0], self.weight_track[-1]))
        plt.plot(self.iter_track, self.weight_track)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists(
            "plt/step_size_conv/weight/{}".format(self.start_time)
        ):
            print(
                "Directory {} created".format(
                    "plt/step_size_conv/weight/{}".format(self.start_time)
                )
            )

        os.makedirs(
            "plt/step_size_conv/weight/{}".format(self.start_time),
            exist_ok=True,
        )
        plt.savefig(
            "plt/step_size_conv/weight/{}/{}.png".format(
                self.start_time, len(self.iter_track)
            )
        )
        plt.close()
