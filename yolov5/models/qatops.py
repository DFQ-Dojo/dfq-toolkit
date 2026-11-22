import torch

# import actnn.cpp_extension.backward_func as ext_backward_func
import torch.nn.functional as F
import torch.nn as nn
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np


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
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1

        assert scale.min() > 0, "step size = {:.6f} becomes non-positive".format(scale)
        times = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, scale)
        ctx.other = times, Qn, Qp

        q_w = (input / scale).round().clamp(Qn, Qp)
        w_q = q_w * scale

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale = ctx.saved_tensors
        times, Qn, Qp = ctx.other

        q_w = input_ / scale

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big  # this is more cpu-friendly than torch.ones(input_.shape)

        grad_scale = (
            ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_output * times)
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
        assert scale.min() > 0, "step size = {:.6f} becomes non-positive".format(scale)
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
        indicate_middle = 1.0 - indicate_small - indicate_big  # this is more cpu-friendly than torch.ones(input_.shape)

        grad_scale = (
            ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_output * times)
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
observe_name = [
    "model.2.m.0.cv1.conv",
    "model.10.conv",
    "model.8.m.0.cv2.conv",
    "model.17.m.0.cv1.conv",
]


def observe_grad(grad: torch.Tensor, name, step, time, max=True, min=True, clip_alpha=1, plot=True):
    clip_value = int(clip_alpha)
    assert min and max, "you should calculate both min and max value"
    grad_np = grad.cpu().detach().numpy()
    flat_grad = grad_np.flatten()
    if min:
        # min_threshold = np.percentile(flat_grad, clip_alpha)
        min_threshold = np.partition(flat_grad, clip_value)[clip_value]
        min_grad = flat_grad[flat_grad <= min_threshold]
    if max:
        max_threshold = np.partition(flat_grad, flat_grad.size - clip_value)[flat_grad.size - clip_value]
        # max_threshold = np.percentile(flat_grad, 100-clip_alpha)
        max_grad = flat_grad[flat_grad >= max_threshold]
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(flat_grad, bins=2000, alpha=0.7, color="green", log=True)
        # plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists("plt2/{}/grad/act/{}".format(time, name)):
            print("Directory {} created".format("plt2/{}/grad/act/{}".format(time, name)))

        os.makedirs("plt2/{}/grad/act/{}".format(time, name), exist_ok=True)
        # if step > 100:
        #     ckpt = {'grad': grad.cpu().detach()}
        #     torch.save(ckpt, "plt/grad/act/{}/{}.pt".format(name,step))
        plt.savefig("plt2/{}/grad/act/{}/{}.png".format(time, name, step))

        plt.hist(min_grad, bins=1000, alpha=0.7, color="blue", log=True)
        plt.savefig("plt2/{}/grad/act/{}/{}_min.png".format(time, name, step))

        plt.hist(max_grad, bins=1000, alpha=0.7, color="purple", log=True)
        plt.savefig("plt2/{}/grad/act/{}/{}_max.png".format(time, name, step))
        plt.close()
    return min_threshold.item(), max_threshold.item()


# def observe_grad(grad: torch.Tensor, name, step, time, plot=True):
#     assert min and max, 'you should calculate both min and max value'
#     grad_np = grad.cpu().detach().numpy()
#     flat_grad = grad_np.flatten()
#     grad_mean = np.mean(flat_grad, dtype=np.float32)
#     grad_std = np.std(flat_grad, dtype=np.float32)
#     min_threshold = grad_mean - 3 * grad_std
#     min_grad = flat_grad[flat_grad <= min_threshold]
#     max_threshold = grad_mean + 3 * grad_std
#     max_grad = flat_grad[flat_grad >= max_threshold]
#     print(f'layer {name}: max threshold is {max_threshold}, min threshold is {min_threshold}')
#     # import IPython
#     # IPython.embed()
#     if plot:
#         plt.figure(figsize=(10, 6))
#         plt.hist(flat_grad, bins=2000, alpha=0.7, color='green', log=True)

#         if not os.path.exists("plt2/{}/grad/act/{}".format(time,name)):
#             print("Directory {} created".format("plt2/{}/grad/act/{}".format(time,name)))

#         os.makedirs("plt2/{}/grad/act/{}".format(time,name), exist_ok=True)
#         plt.savefig("plt2/{}/grad/act/{}/{}.png".format(time,name,step))

#         plt.hist(min_grad, bins=1000, alpha=0.7, color='blue', log=True)
#         plt.savefig("plt2/{}/grad/act/{}/{}_min.png".format(time,name,step))

#         plt.hist(max_grad, bins=1000, alpha=0.7, color='purple', log=True)
#         plt.savefig("plt2/{}/grad/act/{}/{}_max.png".format(time,name,step))
#         plt.close()
#     return min_threshold.item(), max_threshold.item()


def observe_outlier(grad: torch.Tensor, clip_alpha: float):
    clip_value = int(clip_alpha)
    flat_grad = grad.flatten()
    # k_min = int(clip_alpha / 100 * flat_grad.numel())
    k_min = clip_value
    min_threshold, _ = torch.topk(flat_grad, k_min, largest=False)
    min_threshold = min_threshold[-1]
    # k_max = int((1 - clip_alpha / 100)* flat_grad.numel())
    k_max = flat_grad.numel() - clip_value
    max_threshold, _ = torch.topk(flat_grad, k_max, largest=False)
    max_threshold = max_threshold[-1]
    return min_threshold.item(), max_threshold.item()


# def observe_outlier(grad: torch.Tensor):
#     flat_grad = grad.flatten()
#     grad_mean = torch.mean(flat_grad.to(torch.float32))
#     grad_std = torch.std(flat_grad.to(torch.float32))

#     return (grad_mean + 3 * grad_std).item(), (grad_mean - 3 * grad_std).item()


class gradient_clip(torch.autograd.Function):
    """
    used to clip out-of-distribution values in gradient
    """

    @staticmethod
    def forward(ctx, input, name, step, clip, time):
        ctx.saved = name, step, clip, time
        return input

    @staticmethod
    def backward(ctx, grad_output):
        name, step, clip, time = ctx.saved
        if clip > 1e-8:
            if step % 200 == 50 and name in observe_name:
                min, max = observe_grad(grad_output, name, step, time=time, clip_alpha=clip, plot=True)
                # min, max = observe_grad(grad_output, name, step, time=time, plot=True)
            else:
                min, max = observe_outlier(grad_output, clip_alpha=clip)
            grad_output[grad_output > max] = 0
            grad_output[grad_output < min] = 0

        # if step == 252:
        #     exit(0)
        return grad_output, None, None, None, None


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
        num_bits=4,
        mode="quantize",
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
        self.num_bits = num_bits
        self.mode = mode
        self.active_track, self.weight_track, self.iter_track = [], [], []
        self.scale_input = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scale_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        assert self.scale_input.requires_grad, "step size needs grad!"
        assert self.scale_weight.requires_grad, "step size needs grad!"
        self.start_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        self.quantize = False
        self.first_pass = False
        self.name = name
        self.clip = 0.0
        if "clip" in self.mode:
            self.clip = float(self.mode.split("clip")[1])

    def start_quantize(self):
        if self.mode == "exact":
            return
        self.quantize = True
        print("start quantize")

    def forward(self, input):
        if self.quantize and self.mode.startswith("quantize"):
            if not self.first_pass:
                Qsym = 2 ** (self.num_bits - 1) - 1
                Qasym = 2**self.num_bits - 1
                self.scale_weight.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qsym) + 1e-10)
                if self.mode.startswith("quantize_sym"):
                    self.scale_input.data.copy_(2 * input.abs().mean() / math.sqrt(Qsym) + 1e-10)
                elif self.mode.startswith("quantize_asym"):
                    self.scale_input.data.copy_(2 * input.abs().mean() / math.sqrt(Qasym) + 1e-10)
                else:
                    raise NotImplementedError
                print(
                    f"Actually Using LSQconv2d! Init scale_input = {self.scale_input.data.item()} Init scale_weight = {self.scale_weight.data.item()}"
                )
                self.first_pass = True
            else:
                self.scale_input.data = self.scale_input.data.abs()
                self.scale_weight.data = self.scale_weight.data.abs()

            if self.first_pass:
                self.active_track.append(self.scale_input.cpu().detach().numpy())
                self.weight_track.append(self.scale_weight.cpu().detach().numpy())
                self.iter_track.append(len(self.iter_track))
                # if len(self.iter_track) % 20000 == 5000:
                #     self.draw_clip_value()

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
        elif self.mode.startswith("quantize_sym"):
            # return gradient_clip.apply(F.conv2d(SymLsqQuantizer.apply(input, self.scale_input, self.num_bits), SymLsqQuantizer.apply(self.weight, self.scale_weight, self.num_bits), self.bias, self.stride, self.padding, self.dilation, self.groups), self.name, len(self.iter_track), self.clip, self.start_time)
            return F.conv2d(
                SymLsqQuantizer.apply(input, self.scale_input, self.num_bits),
                SymLsqQuantizer.apply(self.weight, self.scale_weight, self.num_bits),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.mode.startswith("quantize_asym"):
            return F.conv2d(
                AsymLsqQuantizer.apply(input, self.scale_input, self.num_bits),
                SymLsqQuantizer.apply(self.weight, self.scale_weight, self.num_bits),
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

        if not os.path.exists("plt/step_size_conv/input/{}".format(self.start_time)):
            print("Directory {} created".format("plt/step_size_conv/input/{}".format(self.start_time)))

        os.makedirs("plt/step_size_conv/input/{}".format(self.start_time), exist_ok=True)
        plt.savefig("plt/step_size_conv/input/{}/{}.png".format(self.start_time, len(self.iter_track)))
        plt.close()

        plt.figure()
        plt.title("{}\n{}".format(self.weight_track[0], self.weight_track[-1]))
        plt.plot(self.iter_track, self.weight_track)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists("plt/step_size_conv/weight/{}".format(self.start_time)):
            print("Directory {} created".format("plt/step_size_conv/weight/{}".format(self.start_time)))

        os.makedirs("plt/step_size_conv/weight/{}".format(self.start_time), exist_ok=True)
        plt.savefig("plt/step_size_conv/weight/{}/{}.png".format(self.start_time, len(self.iter_track)))
        plt.close()
