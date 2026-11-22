import copy
import unittest

import torch

from dfq.detection import Targets, generation_mode
from dfq.distill import DistillConfig, distill_losses
from dfq.models.maskrcnn.detector import MaskRCNNDetector, box_masks
from dfq.models.maskrcnn.resnet import resnet50_detector, resnet50_mask_rcnn
from dfq.quant import Method, QuantConfig, quantize
from dfq.synthesis.generate import PriorWeights, step_terms


def small_batch() -> tuple[torch.Tensor, list[Targets]]:
    images = torch.rand(2, 3, 128, 128)
    targets = [
        Targets(torch.tensor([[10.0, 20.0, 70.0, 90.0]]), torch.tensor([0])),
        Targets(torch.tensor([[30.0, 30.0, 100.0, 120.0]]), torch.tensor([79])),
    ]
    return images, targets


class ResNetMaskRCNNTest(unittest.TestCase):
    detector: MaskRCNNDetector

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.detector = resnet50_detector(resnet50_mask_rcnn())

    def test_box_masks(self) -> None:
        masks = box_masks(torch.tensor([[1.0, 2.0, 4.0, 3.0]]), height=5, width=6)
        self.assertEqual(int(masks.sum()), 3)
        self.assertEqual(masks[0, 2, 1:4].tolist(), [1, 1, 1])

    def test_losses_and_predictions(self) -> None:
        images, targets = small_batch()
        self.detector.network.train()
        losses = self.detector.losses(images, targets)
        self.assertIn("loss_mask", losses)
        self.detector.network.eval()
        with torch.no_grad():
            detections = self.detector.predict(images)
        self.assertEqual(len(detections), 2)
        self.assertTrue(all(bool((d.labels < 80).all()) for d in detections))

    def test_quant_scope_matches_the_old_rule(self) -> None:
        network = copy.deepcopy(self.detector.network)
        config = QuantConfig(method=Method.LSQ, weight_bits=4, activation_bits=4)
        names = quantize(network, config, self.detector.quant_scope())
        # 3 + 4 + 6 + 3 bottlenecks with 3 convs, 4 downsample convs, minus the last bottleneck.
        self.assertEqual(len(names), 16 * 3 + 4 - 3)
        self.assertNotIn("backbone.body.conv1", names)
        self.assertNotIn("backbone.body.layer4.2.conv1", names)

    def test_distillation_and_generation_steps(self) -> None:
        teacher = self.detector
        student = copy.deepcopy(teacher)
        config = QuantConfig(method=Method.LSQ, weight_bits=8, activation_bits=8)
        quantize(student.network, config, student.quant_scope())
        images, targets = small_batch()
        student.network.train()
        terms = distill_losses(
            student, teacher, images, targets, DistillConfig(feature_kind="block")
        )
        self.assertEqual(len(teacher.logits((["a"] * 5, ["b"] * 5))), 10)
        self.assertTrue(torch.isfinite(terms["loss"]))
        generation_mode(teacher.network)
        weights = PriorWeights(batch_norm=0.01, first_batch_norm=1.0, tv_l2=1e-4)
        prior_terms = step_terms(teacher, images, images, targets, weights)
        self.assertGreater(prior_terms["batch_norm"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
