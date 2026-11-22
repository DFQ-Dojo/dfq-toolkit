import unittest

import torch
from torch import nn

from dfq.baselines.psaq import SwinClassifier, dummy_targets, unfold_order
from dfq.models.maskrcnn.swin import PatchMerging


def official_merging(x: torch.Tensor, norm: nn.LayerNorm, reduction: nn.Linear) -> torch.Tensor:
    """The official Swin patch merging: concatenate the four 2x2 neighbours, normalize, reduce."""
    grid = x
    parts = [grid[:, 0::2, 0::2], grid[:, 1::2, 0::2], grid[:, 0::2, 1::2], grid[:, 1::2, 1::2]]
    joined = torch.cat(parts, dim=-1).flatten(1, 2)
    return reduction(norm(joined))


class PsaqTest(unittest.TestCase):
    def test_reordered_weights_reproduce_official_patch_merging(self) -> None:
        torch.manual_seed(0)
        channels, height, width = 8, 4, 6
        norm = nn.LayerNorm(4 * channels)
        nn.init.normal_(norm.weight)
        nn.init.normal_(norm.bias)
        reduction = nn.Linear(4 * channels, 2 * channels, bias=False)
        x = torch.randn(2, height, width, channels)
        expected = official_merging(x, norm, reduction)
        merging = PatchMerging(channels, 2 * channels)
        merging.norm.weight.data = unfold_order(norm.weight.data)
        merging.norm.bias.data = unfold_order(norm.bias.data)
        merging.reduction.weight.data = unfold_order(reduction.weight.data)
        merged, size = merging(x.reshape(2, height * width, channels), (height, width))
        self.assertEqual(size, (2, 3))
        self.assertTrue(torch.allclose(merged, expected, atol=1e-5))

    def test_classifier_shapes(self) -> None:
        logits = SwinClassifier().eval()(torch.rand(1, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (1, 1000))
        self.assertEqual(dummy_targets(2)[1].labels.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
