"""Run directories: one per script run, holding its config, logs and outputs."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def start_run(directory: Path, config: BaseModel) -> Path:
    """Create `directory` and save the resolved config in it as `config.json`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(config.model_dump_json(indent=2))
    return directory


def write_records(path: Path, records: Iterable[BaseModel]) -> None:
    """Append records to a JSON Lines file."""
    lines = [record.model_dump_json() for record in records]
    with path.open("a") as file:
        file.writelines(line + "\n" for line in lines)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
