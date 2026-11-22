import torch
import torch.nn as nn
from functools import partial

from .kl_div import KLDivergence
from .dist_kd import DIST
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
        ori_loss,
        kd_method="kdt4",
        module="",
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        mse_loss_weight=1.0,
        device=None,
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.device = device

        self._teacher_mse_out = []
        self._student_mse_out = []
        self._teacher_kd_out, self._student_kd_out = None, None

        # init kd loss
        if kd_method.startswith("kdt"):
            tau = float(kd_method.split("+")[0][3:])
            self.kd_loss = KLDivergence(tau)
        elif kd_method.startswith("kd"):
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method.startswith("dist_t"):
            tau = float(kd_method.split("+")[0][6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith("dist"):
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
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

        teacher.eval()

    def __call__(self, x, targets):
        with torch.no_grad():
            self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss, loss_items = self.ori_loss(logits, targets)
        total_loss = ori_loss * self.ori_loss_weight

        # compute kd loss
        # kd_loss = torch.zeros(3, device=self.device)
        # for i, (student_out, teacher_out) in enumerate(zip(self._student_out, self._teacher_out)):
        #     kd_loss[i] += self.kd_loss(student_out, teacher_out)
        if len(self._student_mse_out) > len(self._teacher_mse_out):
            initial_len = len(self._student_mse_out)
            self._student_mse_out = self._student_mse_out[-len(self._teacher_mse_out) :]
            print(f"clip len of student out from {initial_len} to {len(self._student_mse_out)}")

        if self.kd_loss is not None:
            kd_loss, kd_loss_items = self.kd_loss(self._student_kd_out, self._teacher_kd_out[1])
            total_loss += kd_loss * self.kd_loss_weight
            del self._student_kd_out, self._teacher_kd_out
        if self.mse_loss is not None:
            mse_loss, mse_loss_items = self.mse_loss(self._student_mse_out, self._teacher_mse_out)
            total_loss += mse_loss * self.mse_loss_weight
            self._student_mse_out, self._teacher_mse_out = [], []
        #
        # print(f'ori loss is {ori_loss.item()}, mse loss is {mse_loss.item()}, kd loss is {kd_loss.item()}')
        return total_loss, loss_items
        # return total_loss, ori_loss, kd_loss, mse_loss, loss_items

    def _register_forward_hook(self, model, name, teacher=False):
        if name == "":
            # use the output of model
            # print(f'register hook, teacher is {teacher}')
            model.register_forward_hook(partial(self._forward_hook, teacher=teacher, name=""))
        else:
            # module = None
            model.register_forward_hook(partial(self._forward_hook, teacher=teacher, name=""))
            for k, m in model.named_modules():
                if any(isinstance(m, layer) for layer in mse_dict[name]):
                    print(f"layer {k} add hook")
                    m.register_forward_hook(partial(self._forward_hook, teacher=teacher, name="layer" + k))

    def _forward_hook(self, module, input, output, teacher=False, name=""):
        # three output [batch_size, 3, 20, 20, 85]; [batch_size, 3, 40, 40, 85]; [batch_size, 3, 80, 80, 85]
        if name != "":
            if self.mse_loss is not None:
                if teacher:
                    self._teacher_mse_out.append(output)
                else:
                    self._student_mse_out.append(output)
        else:
            if self.kd_loss is not None:
                if teacher:
                    tmp = [out.reshape(out.shape[0], -1) for out in output[1]]
                    self._teacher_kd_out = (None, tmp)
                else:
                    self._student_kd_out = [out.reshape(out.shape[0], -1) for out in output]
