import unittest

import torch
from torch import nn

from dfq.quant import Method, QuantConfig, QuantConv2d, QuantLinear, QuantScope, quantize
from dfq.quant.functions import Rounding, learned_step_quantize, round_values


def quantize_with_grads(x: torch.Tensor, scale: float, qn: int, qp: int):
    x = x.clone().requires_grad_()
    step = torch.tensor([scale], requires_grad=True)
    offset = torch.zeros(1)
    y = learned_step_quantize(x, step, offset, qn, qp, 1.0, Rounding.HALF_EVEN)
    y.sum().backward()
    return y, x.grad, step.grad


class FunctionTest(unittest.TestCase):
    def test_forward_rounds_and_clamps(self) -> None:
        x = torch.tensor([-5.0, -0.26, 0.24, 0.5, 9.0])
        y, _, _ = quantize_with_grads(x, scale=0.5, qn=-4, qp=3)
        self.assertEqual(y.tolist(), [-2.0, -0.5, 0.0, 0.5, 1.5])

    def test_backward_is_straight_through_inside_the_range(self) -> None:
        x = torch.tensor([-5.0, 0.3, 9.0])
        _, grad_x, grad_scale = quantize_with_grads(x, scale=0.5, qn=-4, qp=3)
        self.assertEqual(grad_x.tolist(), [0.0, 1.0, 0.0])
        # below: qn = -4; inside: round(0.6) - 0.6 = 0.4; above: qp = 3.
        self.assertAlmostEqual(grad_scale.item(), -4 + 0.4 + 3, places=6)

    def test_half_away_rounding(self) -> None:
        x = torch.tensor([-2.5, -0.5, 0.5, 2.5])
        self.assertEqual(round_values(x, Rounding.HALF_AWAY).tolist(), [-3.0, -1.0, 1.0, 3.0])
        self.assertEqual(round_values(x, Rounding.HALF_EVEN).tolist(), [-2.0, -0.0, 0.0, 2.0])


def small_model() -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 2))


LSQ_4BIT = QuantConfig(method=Method.LSQ, weight_bits=4, activation_bits=4)


class ScopeTest(unittest.TestCase):
    def test_include_and_exclude(self) -> None:
        model = small_model()
        names = quantize(model, LSQ_4BIT, QuantScope(include=("*",), exclude=("2",)))
        self.assertEqual(names, ["0"])
        self.assertIsInstance(model[0], QuantConv2d)
        self.assertNotIsInstance(model[2], QuantLinear)

    def test_quantized_layer_shares_parameters(self) -> None:
        model = small_model()
        weight = model[2].weight
        quantize(model, LSQ_4BIT, QuantScope(include=("*",)))
        self.assertIs(model[2].weight, weight)

    def test_32_bits_is_float(self) -> None:
        model = small_model()
        x = torch.randn(1, 3, 8, 8)
        expected = model(x)
        config = QuantConfig(method=Method.LSQ_PLUS, weight_bits=32, activation_bits=32)
        quantize(model, config, QuantScope(include=("*",)))
        self.assertTrue(torch.equal(model(x), expected))

    def test_state_dict_keeps_trained_step_sizes(self) -> None:
        trained = small_model()
        quantize(trained, LSQ_4BIT, QuantScope(include=("*",)))
        x = torch.randn(2, 3, 8, 8)
        trained(x)
        fresh = small_model()
        quantize(fresh, LSQ_4BIT, QuantScope(include=("*",)))
        fresh.load_state_dict(trained.state_dict())
        other = torch.randn(2, 3, 8, 8) * 10
        self.assertTrue(torch.equal(fresh(other), trained(other)))


if __name__ == "__main__":
    unittest.main()
