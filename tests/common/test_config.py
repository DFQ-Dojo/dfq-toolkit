import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from dfq.common.config import load_config


class Inner(BaseModel):
    bits: int
    method: str


class Outer(BaseModel):
    epochs: int
    quant: Inner


class ConfigTest(unittest.TestCase):
    def test_overrides_replace_nested_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            path.write_text('epochs = 10\n[quant]\nbits = 8\nmethod = "lsq"\n')
            config = load_config(path, Outer, ["quant.bits=4", 'quant.method="lsq+"'])
        self.assertEqual(config, Outer(epochs=10, quant=Inner(bits=4, method="lsq+")))


if __name__ == "__main__":
    unittest.main()
