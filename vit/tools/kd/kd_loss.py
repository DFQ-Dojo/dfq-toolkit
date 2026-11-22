from typing import List, Tuple

import torch
import torch.nn as nn

from mmdet.models.backbones.swin import SwinBlock
from mmdet_custom.models.quant_swin import LSQSwinBlock
from mmdet_custom.models.vit import Block
from mmdet_custom.models.quant_vit import LSQBlock

from tools.kd.mse import MSE
from tools.kd.kl_div import KLDivergence

mse_dict: dict[str, list] = {
    "block": [SwinBlock, LSQSwinBlock, Block, LSQBlock],
    "layer_norm": [nn.LayerNorm],
    "all": [SwinBlock, LSQSwinBlock, Block, LSQBlock, nn.LayerNorm],
}


class KDLoss:
    """
    kd loss wrapper.
    """

    def __init__(
        self,
        student,
        teacher,
        kd_module: str,
        original_loss_weight: float,
        kd_loss_weight: float,
        mse_loss_weight: float,
    ):
        """
        kd_module: has three possible values
            block --- add MSE hooks on the output of Blocks
            layer_norm --- add MSE hooks on all outputs of nn.LayerNorm
            none --- do nothing ...
        """
        self.student = student
        self.teacher = teacher
        self.ori_loss_weight = original_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.mse_modules = kd_module

        self._teacher_mse_out: list = []
        self._student_mse_out: list = []
        self._teacher_kd_out: list = []
        self._student_kd_out: list = []

        # init kd loss
        self.kd_loss: KLDivergence = KLDivergence()

        self.mse_loss: MSE = MSE()

        teacher.eval()

    def enable_kd(self):
        self._register_hook_teacher()
        self._register_hook_student()

    def __call__(self, ori_loss: torch.Tensor):

        with torch.cuda.amp.autocast():
            loss_items = {}
            total_loss = ori_loss * self.ori_loss_weight
            loss_items["detection_loss"] = total_loss.detach().item()

            # compute kd loss
            kd_loss = self.kd_loss(self._student_kd_out, self._teacher_kd_out)
            # print(f"unweighted kd loss is {kd_loss.item()}")
            total_loss += kd_loss * self.kd_loss_weight
            loss_items["kldiv_loss"] = (
                kd_loss.detach().item() * self.kd_loss_weight
            )
            del self._student_kd_out, self._teacher_kd_out
            self._student_kd_out = []
            self._teacher_kd_out = []

            mse_loss, mse_loss_items = self.mse_loss(
                self._student_mse_out, self._teacher_mse_out
            )
            total_loss += mse_loss * self.mse_loss_weight
            loss_items["mse_loss"] = (
                mse_loss.detach().item() * self.mse_loss_weight
            )
            del mse_loss_items, self._student_mse_out, self._teacher_mse_out
            self._student_mse_out = []
            self._teacher_mse_out = []
            # print(f"unweighted mse loss {mse_loss.item()}")

            # print(f"unweighted ori loss is {ori_loss.item()}")
            loss_items["loss"] = total_loss.detach().item()
            return total_loss, loss_items

    def _register_hook_teacher(self):
        self.teacher.rpn_head.register_forward_hook(self._kl_hook_teacher)
        for k, m in self.teacher.named_modules():
            if any(
                isinstance(m, layer) for layer in mse_dict[self.mse_modules]
            ):
                print(f"layer {k} add hook for teacher")
                m.register_forward_hook(self._mse_hook_teacher)

    def _register_hook_student(self):
        self.student.rpn_head.register_forward_hook(self._kl_hook_student)
        for k, m in self.student.named_modules():
            if any(
                isinstance(m, layer) for layer in mse_dict[self.mse_modules]
            ):
                print(f"layer {k} add hook for student")
                m.register_forward_hook(self._mse_hook_student)

    def _mse_hook_teacher(self, module, input, output):
        self._teacher_mse_out.append(output)

    def _mse_hook_student(self, module, input, output):
        if not module.training:
            return
        self._student_mse_out.append(output)

    def _kl_hook_teacher(
        self,
        module,
        input,
        output: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ):
        # output shape:
        #   (List[Tensor]: cls_scores for five scales,
        #           [B, 3, 192 / score, 232 / score]
        #       List[Tensor]: bbox_pred for five scales
        #           [B, 12, 192 / scale, 232 / scale])
        cls_score, bbox_pred = output
        self._teacher_kd_out = cls_score + bbox_pred

    def _kl_hook_student(self, module, input, output):
        if not module.training:
            return
        cls_score, bbox_pred = output
        self._student_kd_out = cls_score + bbox_pred
