import copy
import functools
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dfq.distill import DistillConfig, distill_losses
from dfq.quant import Method, QuantConfig, quantize
from dfq.training import TrainConfig, detection_objective, train
from dfq.training.ema import averaged_copy, update_average
from dfq.training.loop import EmaConfig
from dfq.training.optim import (
    OptimizerConfig,
    OptimizerName,
    ScheduleConfig,
    ScheduleKind,
    WarmupConfig,
    build_optimizer,
    epoch_factor,
    parameter_groups,
    set_rates,
)
from tests.toy import ToyDetector, toy_batch

SGD = OptimizerConfig(name=OptimizerName.SGD, lr=0.01, momentum=0.9, weight_decay=5e-4)


class OptimTest(unittest.TestCase):
    def test_groups_exempt_biases_and_norms(self) -> None:
        network = ToyDetector().network
        biases, decayed, undecayed = parameter_groups(network, no_decay=("head.*",))
        self.assertEqual(len(biases), 5)
        self.assertEqual(len(decayed), 2)
        self.assertEqual(len(undecayed), 3)

    def test_epoch_factors(self) -> None:
        linear = ScheduleConfig(kind=ScheduleKind.LINEAR, final_factor=0.01)
        self.assertEqual(epoch_factor(linear, 0, 10), 1.0)
        self.assertAlmostEqual(epoch_factor(linear, 10, 10), 0.01)
        cosine = ScheduleConfig(kind=ScheduleKind.COSINE, final_factor=0.01)
        self.assertAlmostEqual(epoch_factor(cosine, 10, 10), 0.01)
        step = ScheduleConfig(kind=ScheduleKind.STEP, milestones=(2, 4), gamma=0.1)
        self.assertAlmostEqual(epoch_factor(step, 4, 10), 0.01)

    def test_warmup_starts_biases_high(self) -> None:
        optimizer = build_optimizer(ToyDetector().network, SGD)
        warmup = WarmupConfig(iterations=10, bias_start_lr=0.1, start_momentum=0.8)
        set_rates(optimizer, SGD, warmup, target_lr=0.01, step=0)
        rates = [group["lr"] for group in optimizer.param_groups]
        self.assertEqual(rates, [0.1, 0.0, 0.0])
        self.assertEqual(optimizer.param_groups[1]["momentum"], 0.8)
        set_rates(optimizer, SGD, warmup, target_lr=0.01, step=10)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)


class EmaTest(unittest.TestCase):
    def test_average_copies_quantizer_state(self) -> None:
        network = ToyDetector().network
        config = QuantConfig(method=Method.LSQ, weight_bits=4, activation_bits=4)
        quantize(network, config, ToyDetector().quant_scope())
        average = averaged_copy(network)
        network(torch.rand(1, 3, 8, 8))
        update_average(average, network, updates=1, decay=0.9999, tau=2000)
        self.assertTrue(average.get_submodule("body.0.input_quantizer").started)


def batches(count: int) -> list:
    torch.manual_seed(0)
    return [toy_batch(batch_size=2) for _ in range(count)]


class LoopTest(unittest.TestCase):
    def test_training_lowers_the_loss_and_saves_weights(self) -> None:
        detector = ToyDetector()
        config = TrainConfig(epochs=5, optimizer=SGD.model_copy(update={"lr": 0.1}))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records = train(
                detector.network,
                detection_objective(detector),
                batches(4),
                config,
                lambda _: 0.0,
                run_dir,
            )
            self.assertTrue((run_dir / "best.pt").exists() and (run_dir / "last.pt").exists())
        self.assertLess(records[-1].losses["loss"], records[0].losses["loss"])

    def test_patience_stops_training(self) -> None:
        detector = ToyDetector()
        config = TrainConfig(epochs=10, optimizer=SGD, patience=2)
        with tempfile.TemporaryDirectory() as directory:
            records = train(
                detector.network,
                detection_objective(detector),
                batches(1),
                config,
                lambda _: 1.0,
                Path(directory),
            )
        self.assertEqual(len(records), 3)

    def test_quantized_distillation_with_ema_and_accumulation(self) -> None:
        teacher = ToyDetector()
        student = copy.deepcopy(teacher)
        quant = QuantConfig(method=Method.LSQ_PLUS, weight_bits=4, activation_bits=4)
        quantize(student.network, quant, student.quant_scope())
        objective = functools.partial(
            distill_losses, student, teacher, config=DistillConfig(feature_kind="conv")
        )
        config = TrainConfig(
            epochs=2,
            optimizer=OptimizerConfig(name=OptimizerName.ADAMW, lr=1e-3, no_decay=("head.*",)),
            warmup=WarmupConfig(iterations=3, start_factor=0.001),
            nominal_batch_size=4,
            grad_clip=10.0,
            ema=EmaConfig(),
        )
        scores: list[float] = []

        def evaluate(network: nn.Module) -> float:
            scores.append(float(network.get_parameter("body.0.weight_quantizer.scale").item()))
            return scores[-1]

        with tempfile.TemporaryDirectory() as directory:
            records = train(
                student.network, objective, batches(4), config, evaluate, Path(directory)
            )
        self.assertEqual({"detection", "logit", "feature", "loss"}, set(records[0].losses))
        self.assertEqual(len(scores), 2)


if __name__ == "__main__":
    unittest.main()
