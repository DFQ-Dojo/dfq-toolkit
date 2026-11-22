import copy
import io
import pickle
import unittest

import torch

from dfq.detection import Targets, generation_mode
from dfq.distill import DistillConfig, KLConfig, distill_losses
from dfq.models.yolo.checkpoints import MODULE_PATHS, remapping_module
from dfq.models.yolo.yolo11 import YoloDetector, yolo11_network
from dfq.quant import Method, QuantConfig, quantize
from dfq.synthesis.generate import PriorWeights, step_terms


def batch() -> tuple[torch.Tensor, list[Targets]]:
    images = torch.rand(2, 3, 128, 160)
    targets = [
        Targets(torch.tensor([[10.0, 20.0, 60.0, 90.0]]), torch.tensor([0])),
        Targets(
            torch.tensor([[30.0, 30.0, 150.0, 120.0], [5.0, 5.0, 25.0, 40.0]]),
            torch.tensor([79, 3]),
        ),
    ]
    return images, targets


class Yolo11Test(unittest.TestCase):
    def test_losses_predict_and_scope(self) -> None:
        torch.manual_seed(0)
        detector = YoloDetector(yolo11_network("n"))
        images, targets = batch()
        detector.network.train()
        loss = detector.losses(images, targets)["yolo"]
        self.assertTrue(torch.isfinite(loss) and loss.item() > 0)
        detector.network.eval()
        with torch.no_grad():
            found = detector.predict(images)
        self.assertEqual(len(found), 2)
        config = QuantConfig(method=Method.LSQ, weight_bits=4, activation_bits=4)
        names = quantize(copy.deepcopy(detector.network), config, detector.quant_scope())
        self.assertFalse(any(n.startswith(("model.0.", "model.23.")) for n in names))
        self.assertTrue(any(n.startswith("model.22.") for n in names))

    def test_distillation_and_generation(self) -> None:
        torch.manual_seed(0)
        teacher = YoloDetector(yolo11_network("n"))
        student = copy.deepcopy(teacher)
        config = QuantConfig(method=Method.LSQ, weight_bits=8, activation_bits=8)
        quantize(student.network, config, student.quant_scope())
        images, targets = batch()
        student.network.train()
        distill = DistillConfig(feature_kind="cnnbn", logit_loss=KLConfig(tau=4.0))
        terms = distill_losses(student, teacher, images, targets, distill)
        self.assertTrue(torch.isfinite(terms["loss"]))
        self.assertGreater(terms["feature"].item(), 0.0)
        generation_mode(teacher.network)
        weights = PriorWeights(batch_norm=0.05, first_batch_norm=5.0, tv_l2=0.002)
        prior = step_terms(teacher, images, images, targets, weights)
        self.assertGreater(prior["first_batch_norm"].item(), 0.0)


class Danger:
    pass


class UnpicklerTest(unittest.TestCase):
    def test_refuses_unknown_classes(self) -> None:
        data = pickle.dumps(Danger())
        unpickler = remapping_module(MODULE_PATHS).Unpickler(io.BytesIO(data))
        with self.assertRaises(pickle.UnpicklingError):
            unpickler.load()


if __name__ == "__main__":
    unittest.main()
