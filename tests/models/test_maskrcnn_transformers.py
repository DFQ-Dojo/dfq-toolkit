import unittest

import torch

from dfq.detection import Targets, generation_mode
from dfq.detection.taps import tap
from dfq.models.maskrcnn.mmdet_parts import convert_state_dict, renamed
from dfq.models.maskrcnn.transformer_rcnn import transformer_detector, transformer_mask_rcnn
from dfq.quant import Method, QuantConfig, quantize
from dfq.synthesis.generate import PriorWeights, step_terms


def batch(size: int) -> tuple[torch.Tensor, list[Targets]]:
    images = torch.rand(1, 3, size, size)
    targets = [Targets(torch.tensor([[8.0, 8.0, size - 8.0, size / 2]]), torch.tensor([5]))]
    return images, targets


class ConverterTest(unittest.TestCase):
    def test_renames(self) -> None:
        self.assertEqual(
            renamed("neck.lateral_convs.2.conv.weight"), "backbone.fpn.inner_blocks.2.0.weight"
        )
        self.assertEqual(
            renamed("roi_head.mask_head.convs.3.conv.bias"), "roi_heads.mask_head.3.0.bias"
        )
        self.assertEqual(
            renamed("backbone.stages.1.blocks.0.attn.w_msa.qkv.weight"),
            "backbone.body.stages.1.blocks.0.attn.w_msa.qkv.weight",
        )

    def test_background_moves_to_index_zero(self) -> None:
        state = {
            "roi_head.bbox_head.fc_cls.bias": torch.arange(81.0),
            "roi_head.bbox_head.fc_reg.bias": torch.ones(320),
            "roi_head.mask_head.conv_logits.bias": torch.ones(80),
        }
        converted = convert_state_dict(state)
        self.assertEqual(
            converted["roi_heads.box_predictor.cls_score.bias"][:2].tolist(), [80.0, 0.0]
        )
        self.assertEqual(
            converted["roi_heads.box_predictor.bbox_pred.bias"][:5].tolist(), [0, 0, 0, 0, 1]
        )
        self.assertEqual(converted["roi_heads.mask_predictor.mask_fcn_logits.bias"].shape[0], 81)


class TransformerMaskRCNNTest(unittest.TestCase):
    def check_variant(self, variant: str, quantized_layers: int) -> None:
        torch.manual_seed(0)
        detector = transformer_detector(transformer_mask_rcnn(variant), variant)
        images, targets = batch(224)
        generation_mode(detector.network)
        attention = detector.feature_layers("attention")
        weights = PriorWeights(attention_entropy=1.0)
        terms = step_terms(detector, images, images, targets, weights)
        self.assertTrue(torch.isfinite(terms["attention_entropy"]))
        with tap(detector.network, attention[:1]) as outputs:
            detector.network.eval()
            with torch.no_grad():
                detector.predict(images)
        self.assertEqual(len(outputs), 1)
        config = QuantConfig(method=Method.LSQ, weight_bits=8, activation_bits=8)
        names = quantize(detector.network, config, detector.quant_scope())
        self.assertEqual(len(names), quantized_layers)

    def test_swin_t(self) -> None:
        # patch embedding + 12 blocks x 4 linear layers + 3 patch mergings
        self.check_variant("swin_t", 1 + 12 * 4 + 3)

    def test_deit_t(self) -> None:
        # patch embedding + 12 blocks x 4 linear layers; the transposed up-convolutions stay float
        self.check_variant("deit_t", 1 + 12 * 4)


if __name__ == "__main__":
    unittest.main()
