import torch
import torch.nn as nn
from functools import partial

from quant.kd_mode import KDMethods, KDModules, KDMethodsKind

from quant.kl_div import KLDivergence  # type: ignore
from quant.dist_kd import DIST  # type: ignore
from quant.mse import MSE  # type: ignore

mse_dict: dict[KDModules, list] = {
    KDModules.CNN: [nn.Conv2d],
    KDModules.BN: [nn.BatchNorm2d],
    KDModules.CNNBN: [nn.BatchNorm2d, nn.Conv2d],
    KDModules.ALL: [nn.Module],
}


class KDLoss:
    """
    The total knowledge distillation loss contains three part:
    (1) original loss of the student model
    (2) the KL divergence between teacher model and student model
    (3) the MSE distance between teacher model and student model

    NOTE: Please call enable_kd to register hooks.
    """

    def __init__(
        self,
        student,
        teacher,
        ori_loss,
        kd_methods: KDMethods,
        device: torch.device,
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = kd_methods.original_loss_weight
        self.kd_loss_weight = kd_methods.kd_loss_weight
        self.mse_loss_weight = kd_methods.mse_loss_weight
        self.mse_modules = kd_methods.kd_module
        self.device = device

        self._teacher_mse_out: list = []
        self._student_mse_out: list = []
        self._teacher_kd_out, self._student_kd_out = None, None

        # init kd loss
        if kd_methods.kd_loss.kind == KDMethodsKind.KL:
            self.kd_loss: KLDivergence | DIST = KLDivergence(
                kd_methods.kd_loss.tau
            )
        elif kd_methods.kd_loss.kind == KDMethodsKind.DIST:
            self.kd_loss = DIST(
                beta=kd_methods.kd_loss.beta,
                gamma=kd_methods.kd_loss.gamma,
                tau=kd_methods.kd_loss.tau,
            )
        else:
            raise NotImplementedError("unknown KD algorithm")

        if kd_methods.mse_loss_weight != 0.0:
            self.mse_loss: MSE | None = MSE()
        else:
            self.mse_loss = None

        teacher.eval()

    def enable_kd(self):
        # register forward hook
        self._register_forward_hook(
            self.student, self.mse_modules, teacher=False
        )
        self._register_forward_hook(
            self.teacher, self.mse_modules, teacher=True
        )

    def __call__(self, batch, preds):
        with torch.no_grad():
            tloss, tloss_item = self.teacher(batch["img"])
            del tloss, tloss_item

        # compute ori loss of student
        ori_loss, loss_items = self.ori_loss(batch, preds)
        total_loss = ori_loss * self.ori_loss_weight

        # compute kd loss
        if self.kd_loss is not None:
            kd_loss, kd_loss_items = self.kd_loss(
                self._student_kd_out, self._teacher_kd_out[1]
            )
            total_loss += kd_loss * self.kd_loss_weight
            del self._student_kd_out, self._teacher_kd_out, kd_loss_items
        if self.mse_loss is not None:
            mse_loss, mse_loss_items = self.mse_loss(
                self._student_mse_out, self._teacher_mse_out
            )
            total_loss += mse_loss * self.mse_loss_weight
            del mse_loss_items
            del self._student_mse_out, self._teacher_mse_out
            self._student_mse_out, self._teacher_mse_out = [], []
            print(f"unweighted mse loss {mse_loss.item()}")
        print(
            f"unweighted ori loss is {ori_loss.item()}, unweighted kd loss is {kd_loss.item()}"
        )
        return total_loss, loss_items
        # return total_loss, ori_loss, kd_loss, mse_loss, loss_items

    def _register_forward_hook(
        self, model, name: KDModules, teacher: bool = False
    ):
        # use the output of model
        print(f"register hook, teacher is {teacher}")
        model.register_forward_hook(
            partial(self._forward_hook, teacher=teacher, name="")
        )
        if name != KDModules.NONE:
            for k, m in model.named_modules():
                if (
                    any(isinstance(m, layer) for layer in mse_dict[name])
                    and "dfl" not in k  # DFL is freeze during training
                ):
                    print(f"layer {k} add hook")
                    m.register_forward_hook(
                        partial(
                            self._forward_hook,
                            teacher=teacher,
                            name="layer" + k,
                        )
                    )

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
                    self._student_kd_out = [
                        out.reshape(out.shape[0], -1) for out in output
                    ]
