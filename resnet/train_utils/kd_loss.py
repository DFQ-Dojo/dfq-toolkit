import torch
import torch.nn as nn
from functools import partial
from .loggers import loggers

from .kl_div import KLDivergence
from .mse import MSE

mse_dict = {
    "cnn": [nn.Conv2d],
    "bn": [nn.BatchNorm2d],
    "cnnbn": [nn.BatchNorm2d, nn.Conv2d],
    "all": [nn.Module],
}


class KDLoss:
    """
    kd loss wrapper.
    """

    def __init__(
        self,
        student,
        teacher,
        kd_method="kdt4",
        module="",
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        mse_loss_weight=1.0,
        device=None,
        distill=False,
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.device = device
        self.distill = distill

        self.teacher_mse_out = []
        self.student_mse_out = []
        # self._teacher_mse_debug, self._student_mse_debug = [], []
        self._teacher_kd_out, self._student_kd_out = None, None

        # init kd loss
        if kd_method.startswith("kdt"):
            tau = float(kd_method.split("+")[0][3:])
            self.kd_loss = KLDivergence(tau)
        elif kd_method.startswith("kd"):
            self.kd_loss = KLDivergence()
        else:
            # raise RuntimeError(f'KD method {kd_method} not found.')
            self.kd_loss = None

        if "mse" in kd_method:
            self.mse_loss = MSE()
        else:
            self.mse_loss = None
        # register forward hook
        self._register_forward_hook(student, module, teacher=False)
        self._register_forward_hook(teacher, module, teacher=True)
        loggers["student"].info(
            f"{'*' * 10} Architecture: {'*' * 10}\n{student}\n\n\n{'*' * 10} Layer: {'*' * 10}"
        )
        loggers["teacher"].info(
            f"{'*' * 10} Architecture: {'*' * 10}\n{teacher}\n\n\n{'*' * 10} Layer: {'*' * 10}"
        )

        teacher.train()
        # exit(0)

    def __call__(self, x, targets, original_image_sizes):
        with torch.no_grad():
            self.teacher(x, targets, original_image_sizes)

        # compute ori loss of student
        if self.distill:
            _, ori_loss_dict = self.student(x, targets, original_image_sizes)
        else:
            ori_loss_dict = self.student(x, targets, original_image_sizes)
        # print('have a look on loss dict')
        # import IPython
        # IPython.embed()
        ori_loss = sum(loss for loss in ori_loss_dict.values())
        total_loss = ori_loss * self.ori_loss_weight

        # compute kd loss
        if len(self.student_mse_out) > len(self.teacher_mse_out):
            # import IPython; IPython.embed()
            initial_len = len(self.student_mse_out)
            self.student_mse_out = self.student_mse_out[
                -len(self.teacher_mse_out) :
            ]
            print(
                f"clip len of student out from {initial_len} to {len(self.student_mse_out)}"
            )

        kd_loss = torch.zeros_like(total_loss)
        mse_loss = torch.zeros_like(total_loss)
        if self.kd_loss is not None:
            kd_loss, kd_loss_items = self.kd_loss(
                self._student_kd_out, self._teacher_kd_out
            )
            total_loss += kd_loss * self.kd_loss_weight
            del self._student_kd_out, self._teacher_kd_out
        if self.mse_loss is not None:
            mse_loss, mse_loss_items = self.mse_loss(
                self.student_mse_out, self.teacher_mse_out
            )
            total_loss += mse_loss * self.mse_loss_weight
            self.student_mse_out, self.teacher_mse_out = [], []
        #
        # print(f'ori loss is {ori_loss.item()* self.ori_loss_weight}, mse loss is {mse_loss.item() * self.mse_loss_weight}, kd loss is {kd_loss.item() * self.kd_loss_weight}')
        return (
            total_loss,
            ori_loss * self.ori_loss_weight,
            mse_loss * self.mse_loss_weight,
            kd_loss * self.kd_loss_weight,
        )
        # return total_loss, ori_loss, kd_loss, mse_loss, loss_items

    def _register_forward_hook(self, model, name, teacher=False):
        if name == "":
            # use the output of model
            # print(f'register hook, teacher is {teacher}')
            model.register_forward_hook(
                partial(self._forward_hook, teacher=teacher, name="kd")
            )
        else:
            # module = None
            model.register_forward_hook(
                partial(self._forward_hook, teacher=teacher, name="kd")
            )
            for k, m in model.named_modules():
                if any(isinstance(m, layer) for layer in mse_dict[name]):
                    print(f"layer {k} add hook")
                    m.register_forward_hook(
                        partial(
                            self._forward_hook, teacher=teacher, name="mse " + k
                        )
                    )

    def _recursive_len(self, input):
        if isinstance(input, tuple):
            return (self._recursive_len(item) for item in input)
        elif isinstance(input, list):
            return [self._recursive_len(item) for item in input]
        else:
            return input.shape

    def _forward_hook(self, module, input, output, teacher=False, name=""):
        # three output [batch_size, 3, 20, 20, 85]; [batch_size, 3, 40, 40, 85]; [batch_size, 3, 80, 80, 85]
        if module.training:  # 只在模型的训练阶段发挥作用
            if name != "kd":
                if self.mse_loss is not None:
                    # print('have a look at mse')
                    # import IPython; IPython.embed()
                    if teacher:
                        self.teacher_mse_out.append(output)
                        loggers["teacher"].info(
                            f"{name}, output shape is {self._recursive_len(output)}"
                        )
                        # self._teacher_mse_debug.append(name)
                    else:
                        self.student_mse_out.append(output)
                        loggers["student"].info(
                            f"{name}, output shape is {self._recursive_len(output)}"
                        )
                        # self._student_mse_debug.append(name)
            else:
                # actor = 'teacher' if teacher else 'student'
                # print(f'look at output of {actor} model')
                # import IPython; IPython.embed()
                if self.kd_loss is not None:
                    if teacher:
                        self._teacher_kd_out = [
                            out["scores"].unsqueeze(0) for out in output[0]
                        ]
                        loggers["teacher"].info(
                            f"{name}, output shape is {self._recursive_len([out['scores'] for out in output[0]])}\n\n"
                        )
                    else:
                        self._student_kd_out = [
                            out["scores"].unsqueeze(0) for out in output[0]
                        ]
                        loggers["student"].info(
                            f"{name}, output shape is {self._recursive_len([out['scores'] for out in output[0]])}\n\n"
                        )
