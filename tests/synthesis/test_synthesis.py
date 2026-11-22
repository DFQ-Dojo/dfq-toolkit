import unittest

import torch
from torch import nn

from dfq.detection import Detections, Targets
from dfq.synthesis import BoxSamplingConfig, GenerationConfig, generate_batch
from dfq.synthesis.augment import flip_targets, shift_targets
from dfq.synthesis.box_sampling import resample_image
from dfq.synthesis.generate import LatentSpace, Schedule, learning_rate, to_latent, to_pixels
from dfq.synthesis.priors import (
    batch_norm_statistics,
    gaussian_density,
    statistics_distance,
    total_variation,
)
from tests.toy import ToyDetector, toy_batch


class PriorTest(unittest.TestCase):
    def test_constant_image_has_no_variation(self) -> None:
        l1, l2 = total_variation(torch.ones(2, 3, 5, 5))
        self.assertEqual((l1.item(), l2.item()), (0.0, 0.0))

    def test_matching_statistics_have_no_distance(self) -> None:
        layer = nn.BatchNorm2d(2)
        features = torch.randn(4, 2, 3, 3)
        layer.running_mean = features.mean(dim=(0, 2, 3))
        layer.running_var = features.transpose(0, 1).reshape(2, -1).var(dim=1, unbiased=False)
        self.assertAlmostEqual(statistics_distance(layer, features).item(), 0.0, places=6)

    def test_batch_norm_statistics_cover_every_layer(self) -> None:
        network = ToyDetector().network.eval()
        with batch_norm_statistics(network) as distances:
            network(torch.rand(1, 3, 8, 8))
        self.assertEqual(len(distances), 2)

    def test_density_integrates_to_one(self) -> None:
        samples = torch.rand(1, 50)
        points = torch.linspace(-0.2, 1.2, 2000).unsqueeze(0)
        density = gaussian_density(samples, points)
        integral = torch.trapezoid(density, points, dim=-1)
        self.assertAlmostEqual(integral.item(), 1.0, places=3)


class AugmentTest(unittest.TestCase):
    def test_shift_clips_and_drops_boxes(self) -> None:
        targets = Targets(
            torch.tensor([[0.0, 0.0, 4.0, 4.0], [6.0, 6.0, 9.0, 9.0]]), torch.tensor([0, 1])
        )
        shifted = shift_targets(targets, dx=-5, dy=0, height=10, width=10)
        self.assertEqual(shifted.boxes.tolist(), [[1.0, 6.0, 4.0, 9.0]])
        self.assertEqual(shifted.labels.tolist(), [1])

    def test_flip_mirrors_boxes(self) -> None:
        targets = Targets(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([0]))
        self.assertEqual(flip_targets(targets, width=10).boxes.tolist(), [[7.0, 2.0, 9.0, 4.0]])


def detections(boxes: list[list[float]], scores: list[float]) -> Detections:
    return Detections(
        torch.tensor(boxes), torch.tensor(scores), torch.zeros(len(boxes), dtype=torch.long)
    )


class BoxSamplingTest(unittest.TestCase):
    config = BoxSamplingConfig(confidence=0.2, overlap_iou=0.35, min_area=0.01, max_area=0.85)

    def test_adds_confident_new_boxes_and_drops_unseen_ones(self) -> None:
        targets = Targets(
            torch.tensor([[0.0, 0.0, 20.0, 20.0], [60.0, 60.0, 80.0, 80.0]]), torch.tensor([3, 4])
        )
        found = detections(
            [[1.0, 1.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0], [30.0, 0.0, 40.0, 5.0]],
            [0.9, 0.8, 0.1],
        )
        resampled = resample_image(targets, found, image_area=100.0 * 100.0, config=self.config)
        self.assertEqual(
            resampled.boxes.tolist(), [[0.0, 0.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]]
        )
        self.assertEqual(resampled.labels.tolist(), [3, 0])

    def test_keeps_the_best_box_when_none_is_seen(self) -> None:
        targets = Targets(
            torch.tensor([[0.0, 0.0, 20.0, 20.0], [50.0, 50.0, 70.0, 70.0]]), torch.tensor([3, 4])
        )
        found = detections([[55.0, 55.0, 75.0, 75.0]], [0.9])
        resampled = resample_image(targets, found, image_area=100.0 * 100.0, config=self.config)
        self.assertEqual(resampled.labels.tolist(), [4])


class GenerationTest(unittest.TestCase):
    def test_latent_round_trip(self) -> None:
        space = LatentSpace(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        pixels = torch.rand(1, 3, 4, 4)
        self.assertTrue(
            torch.allclose(to_pixels(to_latent(pixels, space), space), pixels, atol=1e-6)
        )

    def test_schedules(self) -> None:
        cosine = GenerationConfig(iterations=10, lr=1.0)
        self.assertAlmostEqual(learning_rate(cosine, 10), 0.0)
        warm = GenerationConfig(iterations=10, lr=1.0, schedule=Schedule.WARMUP_COSINE, warmup=4)
        self.assertEqual([learning_rate(warm, i) for i in range(1, 5)], [0.25, 0.5, 0.75, 1.0])

    def test_generation_lowers_the_loss_and_keeps_pixels_in_range(self) -> None:
        torch.manual_seed(0)
        teacher = ToyDetector()
        _, targets = toy_batch(batch_size=2, size=16)
        config = GenerationConfig(
            iterations=30, lr=0.1, verify_every=5, box_sampling=BoxSamplingConfig(start=10, stop=20)
        )
        batch = generate_batch(teacher, targets, (16, 16), config, verifier=teacher)
        first, last = batch.history[0].terms["loss"], batch.history[-1].terms["loss"]
        self.assertLess(last, first)
        self.assertGreaterEqual(batch.images.min().item(), 0.0)
        self.assertLessEqual(batch.images.max().item(), 1.0)
        self.assertEqual(batch.images.shape, (2, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
