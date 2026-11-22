import copy
import unittest

import torch

from dfq.detection import Targets, generation_mode
from dfq.distill import DistillConfig, KLConfig, distill_losses
from dfq.models.yolo.yolov5 import Yolov5Detector, scaled_gains, target_rows, yolov5_network
from dfq.quant import Method, QuantConfig, quantize
from dfq.synthesis.generate import PriorWeights, step_terms


def batch() -> tuple[torch.Tensor, list[Targets]]:
    images = torch.rand(2, 3, 128, 160)
    targets = [
        Targets(torch.tensor([[10.0, 20.0, 60.0, 90.0]]), torch.tensor([0])),
        Targets(torch.tensor([[30.0, 30.0, 150.0, 120.0]]), torch.tensor([79])),
    ]
    return images, targets


class Yolov5Test(unittest.TestCase):
    def test_gain_scaling(self) -> None:
        gains = scaled_gains(
            {"box": 0.05, "cls": 0.5, "obj": 1.0}, levels=3, classes=80, image_size=160
        )
        self.assertAlmostEqual(gains["obj"], 1 / 16)
        self.assertAlmostEqual(gains["box"], 0.05)

    def test_target_rows_are_normalized_centres(self) -> None:
        images, targets = batch()
        rows = target_rows(images, targets)
        self.assertEqual(rows[0].tolist(), [0.0, 0.0, 35 / 160, 55 / 128, 50 / 160, 70 / 128])

    def test_losses_predict_distill_and_generate(self) -> None:
        torch.manual_seed(0)
        teacher = Yolov5Detector(yolov5_network("n"), image_size=128)
        student = copy.deepcopy(teacher)
        config = QuantConfig(method=Method.LSQ, weight_bits=8, activation_bits=8)
        names = quantize(student.network, config, student.quant_scope())
        self.assertFalse(any(n.startswith(("model.0.", "model.24.")) for n in names))
        images, targets = batch()
        student.network.train()
        distill = DistillConfig(feature_kind="cnnbn", logit_loss=KLConfig(tau=4.0))
        terms = distill_losses(student, teacher, images, targets, distill)
        self.assertTrue(torch.isfinite(terms["loss"]))
        teacher.network.eval()
        with torch.no_grad():
            self.assertEqual(len(teacher.predict(images)), 2)
        generation_mode(teacher.network)
        prior = step_terms(
            teacher, images, images, targets, PriorWeights(batch_norm=0.01, first_batch_norm=2.0)
        )
        self.assertGreater(prior["batch_norm"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
