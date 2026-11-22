import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        # 根据论文里LEARNED STEP SIZE QUANTIZATION第2节的公式
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        # #根据论文里LEARNED STEP SIZE QUANTIZATION第2.1节
        # #分为三部分：位于量化区间的、小于下界的、大于上界的
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()  # bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float()  # bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        if per_channel:
            grad_alpha = (smaller * Qn + bigger * Qp + between * Round.apply(q_w) - between * q_w) * grad_weight * g
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = (
                ((smaller * Qn + bigger * Qp + between * Round.apply(q_w) - between * q_w) * grad_weight * g)
                .sum()
                .unsqueeze(dim=0)
            )  # ?
        # 在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


# symetric
def calcScale(min_val, max_val, num_bits=8):
    qmin = -(2.0 ** (num_bits - 1))
    qmax = 2.0 ** (num_bits - 1) - 1
    scale = torch.max(abs(min_val), abs(max_val)) / qmax

    return scale


# A(特征)量化
# mode: corresponds to ways of initialization, lsq and minimax
class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init=20, mode="lsq"):
        # activations 没有per-channel这个选项的
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (self.a_bits - 1))
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.g = 1
        self.init_state = 0
        self.quantize = False
        self.calibration = True
        self.mode = mode
        if self.mode == "minimax":
            # min = torch.tensor([], requires_grad=False)
            # max = torch.tensor([], requires_grad=False)
            min = torch.tensor(0)
            max = torch.tensor(0)
            self.register_buffer("min", min)
            self.register_buffer("max", max)

    def start_quantize(self):
        self.quantize = True

    # todo: 在DDP模式下会有些问题
    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        new_scale = calcScale(self.min, self.max, self.a_bits)
        self.s.data.copy_(new_scale)
        # self.s.data.copy_(abs(self.s.data))

    # 量化/反量化
    def forward(self, activation):
        """
                For this work, each layer of weights and each layer of activations has a distinct step size, represented
        as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
        batch of activations, respectively
        """
        # if not self.calibration:
        #     print(f'activation needs grad: {activation.requires_grad}')
        # V1
        if self.quantize:
            with torch.no_grad():
                if self.mode == "lsq":
                    if self.init_state == 0:
                        self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
                        new_s_data = torch.mean(torch.abs(activation.detach())) * 2 / (math.sqrt(self.Qp))
                        self.s.data.copy_(new_s_data)
                        self.init_state += 1
                        print(f"initial activation scale to {self.s.data}")
                    elif self.init_state < self.batch_init:
                        # elif self.calibration:
                        new_s_data = 0.9 * self.s.data + 0.1 * torch.mean(torch.abs(activation.detach())) * 2 / (
                            math.sqrt(self.Qp)
                        )
                        self.s.data.copy_(new_s_data)
                        self.init_state += 1
                    elif self.init_state == self.batch_init:
                        # else:
                        self.s.data.copy_(self.s.data.abs())
                elif self.mode == "minimax":
                    if self.init_state == 0:
                        self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
                    if self.init_state < self.batch_init:
                        self.update(activation.clone())
                        # self.s.data.copy_(abs(torch.max(activation)))
                        self.init_state += 1
                    else:
                        self.s.data.copy_(self.s.data.abs())
                else:
                    raise NotImplementedError
        if self.a_bits == 32:
            output = activation
        elif self.a_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.a_bits != 1
        else:
            try:
                q_a = FunLSQ.apply(activation, self.s, self.g, self.Qn, self.Qp)
            except RuntimeError:
                print("have a look")
                import IPython

                IPython.embed()

        return q_a


# W(权重)量化
class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init=20, mode="lsq"):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (w_bits - 1))
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.g = 1
        self.init_state = 0
        self.quantize = False
        self.calibration = True
        self.mode = mode
        if self.mode == "minimax":
            # min = torch.tensor([], requires_grad=False)
            # max = torch.tensor([], requires_grad=False)
            min = torch.tensor(0)
            max = torch.tensor(0)
            self.register_buffer("min", min)
            self.register_buffer("max", max)

    def start_quantize(self):
        self.quantize = True

    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        new_scale = calcScale(self.min, self.max, self.w_bits)
        self.s.data = new_scale

    # 量化/反量化
    def forward(self, weight):
        if self.quantize:
            with torch.no_grad():
                if self.mode == "lsq":
                    if self.init_state == 0:
                        self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
                        if self.per_channel:
                            weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                            new_s_data = torch.mean(torch.abs(weight_tmp), dim=1) * 2 / (math.sqrt(self.Qp))
                            self.s.data.copy_(new_s_data)
                        else:
                            new_s_data = torch.mean(torch.abs(weight.detach())) * 2 / (math.sqrt(self.Qp))
                            self.s.data.copy_(new_s_data)
                        self.init_state += 1
                        print(f"initial weight scale to {self.s.data}")
                    elif self.init_state < self.batch_init:
                        # elif self.calibration:
                        if self.per_channel:
                            weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                            new_s_data = 0.9 * self.s.data + 0.1 * torch.mean(torch.abs(weight_tmp), dim=1) * 2 / (
                                math.sqrt(self.Qp)
                            )
                            self.s.data.copy_(new_s_data)
                        else:
                            new_s_data = 0.9 * self.s.data + 0.1 * torch.mean(torch.abs(weight.detach())) * 2 / (
                                math.sqrt(self.Qp)
                            )
                            self.s.data.copy_(new_s_data)
                        self.init_state += 1
                    elif self.init_state == self.batch_init:
                        # else:
                        self.s.data.copy_(self.s.data.abs())
                elif self.mode == "minimax":
                    assert not self.per_channel, "minimax initialization does not support per channel quantization"
                    if self.init_state == 0:
                        self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
                    # if self.calibration:
                    if self.init_state < self.batch_init:
                        self.update(weight.data)
                        # self.s.data.copy_(abs(torch.max(weight)))
                        self.init_state += 1
                    else:
                        self.s.data.copy_(self.s.data.abs())
                else:
                    raise NotImplementedError
        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.w_bits != 1
        else:
            # print(self.s, self.g)
            try:
                w_q = FunLSQ.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)
            except RuntimeError:
                print("have a look on weight")
                import IPython

                IPython.embed()

            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q


class QuantConv2d(nn.Conv2d):
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
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=20,  # num of samples
        initialize_mode="minimax",
    ):
        super(QuantConv2d, self).__init__(
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
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(
            a_bits=a_bits,
            all_positive=all_positive,
            batch_init=batch_init,
            mode=initialize_mode,
        )
        self.weight_quantizer = LSQWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
            mode=initialize_mode,
        )
        self.train_batch = 0

    def start_quantize(self):
        self.activation_quantizer.start_quantize()
        self.weight_quantizer.start_quantize()

    def end_calibration(self):
        self.activation_quantizer.calibration = False
        self.weight_quantizer.calibration = False

    def forward(self, input):
        if self.activation_quantizer.quantize and self.train_batch == 0:
            self.train_batch = input.shape[0]
            print(f"batch size is: {self.train_batch}")
            self.activation_quantizer.batch_init = 2000 // self.train_batch + 1
            self.weight_quantizer.batch_init = 2000 // self.train_batch + 1
            # self.activation_quantizer.batch_init = 1
            # self.weight_quantizer.batch_init = 1
        quant_input = self.activation_quantizer(input)
        # print('input:',input.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output


if __name__ == "__main__":
    test = QuantConv2d(1, 1, 1)
    test2 = nn.Conv2d(1, 1, 1)
    print(isinstance(test, nn.Conv2d))
    print(isinstance(test2, nn.Conv2d))
    # import torch.optim as optim

    # class SimpleCNN(nn.Module):
    #     def __init__(self):
    #         super(SimpleCNN, self).__init__()
    #         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #         self.fc = nn.Linear(16 * 16 * 16, 10)

    #     def forward(self, x):
    #         x = self.pool(nn.functional.relu(self.conv1(x)))
    #         x = x.view(-1, 16 * 16 * 16)
    #         x = self.fc(x)
    #         return x

    # model = SimpleCNN()
    # inputs =  torch.randn(1, 3, 32, 32)

    # # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1)

    # # 将梯度缓冲区的梯度归零，并进行前向传播和反向传播
    # optimizer.zero_grad()
    # outputs = model(inputs)
    # targets = torch.randint(0, 10, (1,), dtype=torch.long)   # 随机生成一个目标类别
    # loss = criterion(outputs, targets)
    # loss.backward()

    # # 检查模型参数的梯度张量的数据类型
    # for name, param in model.named_parameters():
    #     if 'conv' in name:  # 只打印卷积层的参数梯度数据类型
    #         print(f"Parameter: {name}, Gradient dtype: {param.grad.dtype}")


# def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
#                  quant_inference=False, all_positive=False, per_channel=False,
#                  batch_init = 20):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Conv2d):
#             layer_counter[0] += 1
#             if layer_counter[0] >= 1: #第一层也量化
#                 if child.bias is not None:
#                     quant_conv = QuantConv2d(child.in_channels, child.out_channels,
#                                              child.kernel_size, stride=child.stride,
#                                              padding=child.padding, dilation=child.dilation,
#                                              groups=child.groups, bias=True, padding_mode=child.padding_mode,
#                                              a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                     quant_conv.bias.data = child.bias
#                 else:
#                     quant_conv = QuantConv2d(child.in_channels, child.out_channels,
#                                              child.kernel_size, stride=child.stride,
#                                              padding=child.padding, dilation=child.dilation,
#                                              groups=child.groups, bias=False, padding_mode=child.padding_mode,
#                                              a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                 quant_conv.weight.data = child.weight
#                 module._modules[name] = quant_conv
#         elif isinstance(child, nn.ConvTranspose2d):
#             layer_counter[0] += 1
#             if layer_counter[0] >= 1: #第一层也量化
#                 if child.bias is not None:
#                     quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
#                                                                 child.out_channels,
#                                                                 child.kernel_size,
#                                                                 stride=child.stride,
#                                                                 padding=child.padding,
#                                                                 output_padding=child.output_padding,
#                                                                 dilation=child.dilation,
#                                                                 groups=child.groups,
#                                                                 bias=True,
#                                                                 padding_mode=child.padding_mode,
#                                                                 a_bits=a_bits,
#                                                                 w_bits=w_bits,
#                                                                 quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                     quant_conv_transpose.bias.data = child.bias
#                 else:
#                     quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
#                                                                 child.out_channels,
#                                                                 child.kernel_size,
#                                                                 stride=child.stride,
#                                                                 padding=child.padding,
#                                                                 output_padding=child.output_padding,
#                                                                 dilation=child.dilation,
#                                                                 groups=child.groups, bias=False,
#                                                                 padding_mode=child.padding_mode,
#                                                                 a_bits=a_bits,
#                                                                 w_bits=w_bits,
#                                                                 quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                 quant_conv_transpose.weight.data = child.weight
#                 module._modules[name] = quant_conv_transpose
#         elif isinstance(child, nn.Linear):
#             layer_counter[0] += 1
#             if layer_counter[0] >= 1: #第一层也量化
#                 if child.bias is not None:
#                     quant_linear = QuantLinear(child.in_features, child.out_features,
#                                                bias=True, a_bits=a_bits, w_bits=w_bits,
#                                                quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                     quant_linear.bias.data = child.bias
#                 else:
#                     quant_linear = QuantLinear(child.in_features, child.out_features,
#                                                bias=False, a_bits=a_bits, w_bits=w_bits,
#                                                quant_inference=quant_inference,
#                                              all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
#                 quant_linear.weight.data = child.weight
#                 module._modules[name] = quant_linear
#         else:
#             add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
#                          quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)


# def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
#             all_positive=False, per_channel=False, batch_init = 20):
#     if not inplace:
#         model = copy.deepcopy(model)
#     layer_counter = [0]
#     add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
#                  quant_inference=quant_inference, all_positive=all_positive,
#                  per_channel=per_channel, batch_init = batch_init)
#     return model
