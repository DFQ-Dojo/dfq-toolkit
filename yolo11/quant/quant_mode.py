from enum import StrEnum

from pydantic import BaseModel, Field

from typing import Annotated, Literal


class QuantizeKind(StrEnum):
    EXACT = "exact"
    SYMMETRIC_QUANTIZE = "quantize_sym"
    ASYMMETRIC_QUANTIZE = "quantize_asym"


class ExactModel(BaseModel):
    """Do not quantize the model."""

    kind: Literal[QuantizeKind.EXACT]


class SymQuantizeOption(BaseModel):
    """
    Use LSQ(Learned Step-size Quantization, https://arxiv.org/abs/1902.08153) to quantize the model.

    weight_bits is the bit-width of network weights.
    activation_bits is the bit-width of intermediate value.
    """

    kind: Literal[QuantizeKind.SYMMETRIC_QUANTIZE]
    weight_bits: int
    activation_bits: int


class AsymQuantizeOption(BaseModel):
    """
    Use LSQ+(https://arxiv.org/abs/2004.09576) to quantize the model.

    weight_bits is the bit-width of network weights.
    activation_bits is the bit-width of intermediate value.
    """

    kind: Literal[QuantizeKind.ASYMMETRIC_QUANTIZE]
    weight_bits: int
    activation_bits: int


# Our code supports three modes of model quantization. They are:
# - ExactModel: Do not quantize the model.
# - SymQuantizeModel: Use LSQ to quantize the model. This is controlled by `SymQuantizeOption` class.
#                     See its definition for more instructions.
# - AsymQuantizeOption: Use LSQ+ to quantize the model. This is controlled by `AsymQuantizeOption` class.
#                     See its definition for more instructions.
QuantizeMode = Annotated[
    ExactModel | SymQuantizeOption | AsymQuantizeOption,
    Field(discriminator="kind"),
]
