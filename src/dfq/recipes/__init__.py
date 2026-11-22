"""Recipes connect detector adapters to the pipelines. Only scripts import this layer.

Each recipe takes one Pydantic config record (read from a TOML file in configs/) and runs one
stage: making labels, generating images, quantization-aware training, or evaluation.
"""
