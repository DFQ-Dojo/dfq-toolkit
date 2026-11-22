import tempfile
import unittest
from pathlib import Path

import torch

from dfq.data.coco import CocoInstances, pad_batch, resize_keep_ratio
from dfq.data.coco_eval import evaluate_coco
from dfq.data.labels import (
    coco_labels,
    from_targets,
    one_box_labels,
    read_label_set,
    to_targets,
    write_label_set,
)
from dfq.data.synthetic import SyntheticImages, append_batch, label_set_of, start_set
from dfq.detection import Detections, Targets
from tests.coco_fixture import write_coco


class LabelTest(unittest.TestCase):
    def test_one_box_labels_stay_inside_the_image(self) -> None:
        for item in one_box_labels(200, 80, seed=0):
            box = item.boxes[0]
            self.assertTrue(0 <= box.x1 < box.x2 <= 1 and 0 <= box.y1 < box.y2 <= 1)

    def test_label_set_round_trip(self) -> None:
        items = one_box_labels(3, 80, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            write_label_set(path, items)
            self.assertEqual(read_label_set(path), items)

    def test_targets_round_trip(self) -> None:
        item = one_box_labels(1, 80, seed=2)[0]
        targets = to_targets(item, height=160, width=320)
        again = from_targets(targets, height=160, width=320, image_id=item.image_id)
        self.assertAlmostEqual(again.boxes[0].x2, item.boxes[0].x2, places=6)

    def test_coco_labels_use_contiguous_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, annotations = write_coco(Path(directory))
            items = coco_labels(annotations)
        self.assertEqual([[b.label for b in item.boxes] for item in items], [[0, 2], [79]])
        self.assertAlmostEqual(items[0].boxes[0].x2, 0.4)


class CocoTest(unittest.TestCase):
    def test_load_resize_and_pad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images, annotations = write_coco(Path(directory))
            dataset = CocoInstances(images, annotations, masks=True, skip_empty=True)
            samples = [resize_keep_ratio(dataset[i], short=160, long=200) for i in range(2)]
        image, targets, info = samples[0]
        self.assertEqual(tuple(image.shape), (3, 160, 200))
        self.assertAlmostEqual(info.scale, 2.0)
        self.assertEqual(targets.boxes[0].tolist(), [20.0, 40.0, 80.0, 120.0])
        assert targets.masks is not None
        self.assertEqual(int(targets.masks[0].sum()), 60 * 80)
        batch, padded, _ = pad_batch(samples, fill=(0.5, 0.5, 0.5))
        self.assertEqual(tuple(batch.shape), (2, 3, 224, 224))
        self.assertEqual(batch[0, 0, 200, 0].item(), 0.5)
        assert padded[1].masks is not None
        self.assertEqual(tuple(padded[1].masks.shape[1:]), (224, 224))


class PerfectDetector:
    """Predicts exactly the ground truth of the batch it was given."""

    def __init__(self) -> None:
        self.network = torch.nn.Linear(1, 1)
        self.num_classes = 80
        self.answers: list[Targets] = []

    def predict(self, images: torch.Tensor) -> list[Detections]:
        return [Detections(t.boxes, torch.ones(len(t.labels)), t.labels) for t in self.answers]


class EvaluationTest(unittest.TestCase):
    def test_perfect_detections_score_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images, annotations = write_coco(Path(directory))
            dataset = CocoInstances(images, annotations, masks=False, skip_empty=False)
            samples = [resize_keep_ratio(dataset[i], short=160, long=200) for i in range(2)]
            batch = pad_batch(samples, fill=(0.0, 0.0, 0.0))
            detector = PerfectDetector()
            detector.answers = batch[1]
            stats = evaluate_coco(detector, [batch], dataset.coco, with_masks=False)  # type: ignore[arg-type]
        self.assertAlmostEqual(stats["bbox_mAP"], 1.0, places=6)


class SyntheticTest(unittest.TestCase):
    def test_append_and_read(self) -> None:
        targets = [
            Targets(torch.tensor([[1.0, 2.0, 8.0, 9.0]]), torch.tensor([5])) for _ in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_set(root, height=16, width=16)
            append_batch(root, torch.rand(3, 3, 16, 16), targets)
            append_batch(root, torch.rand(3, 3, 16, 16), targets)
            dataset = SyntheticImages(root, limit=5)
            image, target = dataset[4]
            labels = label_set_of(root)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(target.labels.tolist(), [5])
        self.assertEqual([item.image_id for item in labels][-1], "000000000005")


if __name__ == "__main__":
    unittest.main()
