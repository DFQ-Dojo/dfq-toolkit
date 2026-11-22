import copy
import unittest

import torch

from dfq.distill import DISTConfig, DistillConfig, KLConfig, distill_losses
from dfq.distill.losses import SoftmaxOver, kl_divergence
from dfq.distill.taps import tap
from dfq.quant import Method, QuantConfig, quantize
from tests.toy import ToyDetector, toy_batch


def student_and_teacher() -> tuple[ToyDetector, ToyDetector]:
    teacher = ToyDetector()
    student = copy.deepcopy(teacher)
    return student, teacher


class DistillTest(unittest.TestCase):
    def test_identical_models_have_no_distillation_loss(self) -> None:
        student, teacher = student_and_teacher()
        student.network.eval()
        images, targets = toy_batch()
        config = DistillConfig(feature_kind="conv")
        terms = distill_losses(student, teacher, images, targets, config)
        self.assertAlmostEqual(terms["logit"].item(), 0.0, places=5)
        self.assertAlmostEqual(terms["feature"].item(), 0.0, places=6)

    def test_quantized_student_gets_gradients(self) -> None:
        student, teacher = student_and_teacher()
        config = QuantConfig(method=Method.LSQ, weight_bits=2, activation_bits=2)
        quantize(student.network, config, student.quant_scope())
        student.network.train()
        images, targets = toy_batch()
        distill = DistillConfig(feature_kind="conv", logit_loss=DISTConfig())
        terms = distill_losses(student, teacher, images, targets, distill)
        terms["loss"].backward()
        self.assertGreater(terms["feature"].item(), 0.0)
        scale = student.network.get_parameter("body.0.weight_quantizer.scale")
        self.assertIsNotNone(scale.grad)
        self.assertTrue(all(p.grad is None for p in teacher.network.parameters()))

    def test_no_feature_layers(self) -> None:
        student, teacher = student_and_teacher()
        images, targets = toy_batch()
        config = DistillConfig(feature_kind="none", logit_loss=KLConfig(tau=4.0))
        terms = distill_losses(student, teacher, images, targets, config)
        self.assertEqual(set(terms), {"detection", "logit", "loss"})

    def test_tap_removes_its_hooks(self) -> None:
        network = ToyDetector().network
        with tap(network, ["head"]) as outputs:
            network(torch.rand(1, 3, 8, 8))
        network(torch.rand(1, 3, 4, 4))
        head_output = outputs["head"]
        assert isinstance(head_output, torch.Tensor)
        self.assertEqual(head_output.shape[-1], 8)
        self.assertFalse(network.get_submodule("head")._forward_hooks)

    def test_joint_softmax_differs_from_per_output(self) -> None:
        student = [torch.randn(2, 5), torch.randn(2, 3)]
        teacher = [torch.randn(2, 5), torch.randn(2, 3)]
        each = kl_divergence(student, teacher, 1.0, SoftmaxOver.EACH_OUTPUT)
        joint = kl_divergence(student, teacher, 1.0, SoftmaxOver.ALL_OUTPUTS)
        self.assertNotAlmostEqual(each.item(), joint.item())


if __name__ == "__main__":
    unittest.main()
