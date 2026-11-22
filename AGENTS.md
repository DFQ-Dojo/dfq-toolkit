This repository holds `dfq`, the package for task-specific zero-shot quantization of object
detectors (YOLOv5, YOLO11, ResNet and Swin/DeiT Mask R-CNN). The design is in
[`docs/design/unified-package.md`](docs/design/unified-package.md); the layers are in
[`docs/architecture.md`](docs/architecture.md).

# Project philosophy

- **The project owns execution.** Our loops generate images and run quantization-aware training.
  Ultralytics, torchvision and timm provide models, losses and layers, not the workflow.
- **Pipelines depend on protocols, adapters implement them.** `synthesis/`, `distill/` and
  `training/` depend on the `Detector` protocol in `detection/`. `models/` implements it. They never
  import each other; only `scripts/` connect them.
- **Stable modules never import fast-changing ones.** Layers, from stable to fast-changing:
  `common/` -> `quant/` -> `detection/` -> `data/`, `distill/`, `synthesis/`, `training/`;
  `models/` sits beside the last four.
- **Use Pydantic for persisted data.** Configs, label files, manifests and results validate and
  round-trip. Runtime collaborators (models, losses, priors) are protocols or ordinary classes.
- **Explicit over implicit.** Follow the Zen of Python. No registries, no import side effects, no
  monkey-patching of a model's methods.
- **Less state.** Prefer functions that take and return values to classes that mutate fields.
- **No pink elephants.** Do not add defensive clauses for cases that cannot happen.
- **Flat calls.** Assign each call's result to a name; avoid `f(g(h(x)))`.
- **Dependency order.** Define a function before the functions that call it.
- **Small commits.** Each commit changes at most about 1000 lines, ideally 500, not counting moved
  files.

# Documentation

A change is not done until its docs are. Update the docs in the same commit.

| Doc | Owns |
|---|---|
| `README.md` | What the project is and where to start |
| `docs/architecture.md` | Layers, modules, protocols, records, data flow, how to add a detector |
| `docs/user-manual.md` | How to install and run every script, and what it writes |
| `docs/design/<topic>.md` | One design: status, decision, rejected option, limitations |
| `docs/research/` | Results: setup, numbers, run directory, date |
| `docs/glossary.md` | Every term and abbreviation |

Write short sentences in plain English, in the active voice. Define each term the first time a doc
uses it. Write dates as absolute dates (2026-11-22).

# Checks

```bash
uv run ruff check && uv run ruff format --check && uv run mypy && uv run python -m unittest
```
