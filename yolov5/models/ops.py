import torch

# import actnn.cpp_extension.backward_func as ext_backward_func
from torch.autograd.function import Function
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
import math
from datetime import datetime


class SymLsqQuantizer(torch.autograd.Function):
    """
    Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def run_forward(input, scale, Qn, Qp):
        """
        :param input: input to be quantized
        :param scale: the step size
        :param num_bits: quantization bits
        :return: quantized output
        """
        assert scale.min() > 0, "alpha = {:.6f} becomes non-positive".format(scale)
        # print(alpha.min())
        # print("-" * 100)
        times = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        # ctx.save_for_backward(input, alpha)

        q_w = (input / scale).round().clamp(Qn, Qp)
        w_q = q_w * scale

        return w_q, times

    @staticmethod
    def run_backward(grad_output, input_, scale, times, Qn, Qp):
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

        return grad_input, grad_scale


# class LsqStepSize(nn.Parameter):
#     def __init__(self, tensor):
#         super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
#         # self.data.requires_grad = True
#         # print(self.data.requires_grad)
#         self.initialized = False

#     def _initialize(self, init_tensor):
#         assert not self.initialized, 'already initialized.'
#         self.data.copy_(init_tensor)
#         # print('Stepsize initialized to %.6f' % self.data.item())
#         self.initialized = True

#     def change_abs(self):
#         if self.data.min() < 0:
#             data_min = self.data.min()
#             self.data = self.data.abs()
#             print("change the min from {} to {}".format(data_min, self.data.min()))


#     def initialize_wrapper(self, tensor, num_bits):
#         # input: everthing needed to initialize step_size
#         Qp = 2 ** (num_bits - 1) - 1
#         init_val = 2 * tensor.abs().mean() / math.sqrt(Qp)

#         eps = 1e-10 * torch.ones_like(init_val)
#         init_val += eps
#         # print("361 tensor {} init val {} self.data {} ".format(tensor.shape, init_val.shape, self.data.shape))
#         self._initialize(init_val)


class convnd(Function):
    @staticmethod
    def run_forward(forward_op, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return forward_op(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def run_backward(ctx, grad_output, bias_reduce_dims, aug):
        stride, padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        stride = aug(stride)
        dilation = aug(dilation)

        input, weight, bias, _, _, _, _, _, _, _, _ = ctx.saved
        # del quantized, ctx.saved

        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1988
        # Backward pass for convolution. Computes gradients for input, weight, and bias depending on the
        # output_mask setting. This function supports 1D, 2D, or 3D spatial convolution and currently requires
        # a single batch dimension to be present.

        # Args:
        # grad_output_: tensor of shape (N, C_out, L_out), (N, C_out, H_out, W_out), or (N, C_out, D_out, H_out, W_out)
        # input_: tensor of shape (N, C_in, L_in), (N, C_in, H_in, W_in), or (N, C_in, D_in, H_in, W_in)
        # weight_: tensor of shape (C_out, C_in // groups, *kernel_size); dimension of kernel_size must match the number
        #     of input spatial dimensions
        # bias_sizes_opt: if specified, indicates that a bias was used in the forward pass and contains the shape
        #     of the bias. While the bias shape can be computed from other inputs, it is provided to this function for
        #     ease of use. The bias shape is (weight.shape[0]) for normal convolution and (weight.shape[1] * groups)
        #     for transposed convolution.
        # stride: single value or an array with dimension matching the number of input spatial dimensions
        # padding: single value or an array with dimension matching the number of input spatial dimensions
        # dilation: single value or an array with dimension matching the number of input spatial dimensions
        # transposed: boolean indicating whether the convolution is transposed
        # output_padding: single value or dimension == number of input spatial dimensions; only supported when
        #     transposed is true
        # groups: number of groups for grouped convolution
        # output_mask: 3-dim boolean array specifying which gradients to compute in input, weight, bias order
        # todo: 有点问题，明天de一下，看看是什么原因
        # grad_input, grad_weight, grad_bias = ext_backward_func.convolution_backward(
        #     grad_output, input, weight, [weight.shape[0]], stride, padding, dilation, False, [0,0], groups,
        #     [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]])
        # try:
        #     assert ctx.needs_input_grad[0] and ctx.needs_input_grad[1]
        # except AssertionError:
        #     import IPython
        #     IPython.embed()
        grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
            grad_output.float(),
            input.float(),
            weight,
            [weight.shape[0]],
            stride,
            padding,
            dilation,
            False,
            [0, 0],
            groups,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]],
        )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(bias_reduce_dims)
        else:
            grad_bias = None

        if grad_input is not None:
            grad_input = grad_input.half()
        return grad_input, grad_weight, grad_bias


class conv_act_lsq(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        scale_input,
        scale_weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_bits=4,
    ):

        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        q_input, times_input = SymLsqQuantizer.run_forward(input, scale_input, Qn, Qp)
        q_weight, times_weight = SymLsqQuantizer.run_forward(weight, scale_weight, Qn, Qp)
        ctx.saved = (
            q_input,
            q_weight,
            bias,
            times_input,
            times_weight,
            scale_input,
            scale_weight,
            input,
            weight,
            Qn,
            Qp,
        )
        ctx.other_args = (stride, padding, dilation, groups)
        return convnd.run_forward(F.conv2d, q_input, q_weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_hidden, grad_weight_hidden, grad_bias = convnd.run_backward(ctx, grad_output, [0, 2, 3], _pair)
        (
            _,
            _,
            _,
            times_input,
            times_weight,
            scale_input,
            scale_weight,
            input,
            weight,
            Qn,
            Qp,
        ) = ctx.saved
        if grad_input_hidden is not None:
            grad_input, grad_scale_input = SymLsqQuantizer.run_backward(
                grad_input_hidden, input, scale_input, times_input, Qn, Qp
            )
        else:
            grad_input, grad_scale_input = None, None
        grad_weight, grad_scale_weight = SymLsqQuantizer.run_backward(
            grad_weight_hidden, weight, scale_weight, times_weight, Qn, Qp
        )
        out = (
            grad_input,
            grad_weight,
            grad_scale_input,
            grad_scale_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )
        # import IPython
        # IPython.embed()
        return out


class conv_act_test(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.saved = input, weight, bias, None, None, None, None, None, None, None, None
        ctx.other_args = (stride, padding, dilation, groups)
        return convnd.run_forward(F.conv2d, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = convnd.run_backward(ctx, grad_output, [0, 2, 3], _pair)
        return grad_input, grad_weight, grad_bias, None, None, None, None


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
        self.scale_input = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scale_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        assert self.scale_input.requires_grad, "step size needs grad!"
        assert self.scale_weight.requires_grad, "step size needs grad!"
        self.num_bits = num_bits
        self.mode = mode
        self.active_track, self.weight_track, self.iter_track = [], [], []
        self.start_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

        self.first_pass = -2

    def draw_clip_value(self):

        import matplotlib.pyplot as plt
        import os

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

    # todo: check是否符合预期
    # todo: 增加记录函数，记录训练过程中step size的变化情况
    def forward(self, input):
        # if self.first_pass == -2: # pass the model.info() function in yolo
        #     self.first_pass += 1
        # self.training and
        if self.mode == "quantize":
            if self.first_pass == -2:
                self.first_pass += 1
            elif self.first_pass >= 0:
                self.scale_input.data = self.scale_input.data.abs()
                self.scale_weight.data = self.scale_weight.data.abs()
            else:
                self.first_pass += 1
                Qp = 2 ** (self.num_bits - 1) - 1
                self.scale_input.data.copy_(2 * input.abs().mean() / math.sqrt(Qp) + 1e-10)
                self.scale_weight.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp) + 1e-10)
                print(
                    f"Actually Using LSQconv2d! Init scale_input = {self.scale_input.data.item()} Init scale_weight = {self.scale_weight.data.item()}"
                )

            if self.first_pass >= 0:
                self.active_track.append(self.scale_input.cpu().detach().numpy())
                self.weight_track.append(self.scale_weight.cpu().detach().numpy())
                self.iter_track.append(len(self.iter_track))
                if len(self.iter_track) % 2000 == 500:
                    self.draw_clip_value()

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
        elif self.mode == "quantize":
            return conv_act_lsq.apply(
                input,
                self.weight,
                self.scale_input,
                self.scale_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.num_bits,
            )
        elif self.mode == "test":
            return conv_act_test.apply(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            raise NotImplementedError


# class LSQconv2d(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
#         return convnd.run_forward(2, F.conv2d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return convnd.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)

if __name__ == "__main__":
    import torch.nn.init as init

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

        def forward(self, x):
            x = self.conv(x)
            return x

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv = LSQconv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                num_bits=8,
                mode="quantize",
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    model = SimpleModel()
    test = TestModel()
    init.constant_(model.conv.weight, 2)
    init.constant_(test.conv.weight, 2)
    init.constant_(model.conv.bias, 1)
    init.constant_(test.conv.bias, 1)

    input_data = nn.Parameter(torch.randn(1, 1, 5, 5), requires_grad=True)  # (batch_size, channels, height, width)

    output = model(input_data)
    testout = test(input_data)
    loss_fn = nn.MSELoss()
    target = torch.randn_like(output)

    loss = loss_fn(output, target)
    test_loss = loss_fn(testout, target)
    loss.backward()
    test_loss.backward()
    print("Conv2d grad:", model.conv.weight.grad)
    print("LSQConv2d grad:", test.conv.weight.grad)
