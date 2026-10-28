"""Type system for the Neutral GPU Intermediate Representation."""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class AddressSpace(Enum):
    """GPU physical memory hierarchies."""
    GLOBAL = "global"
    SHARED = "shared"      # AMD LDS / Vulkan Workgroup / CUDA Shared
    LOCAL = "local"        # Thread-private register stack
    CONSTANT = "constant"  # Uniform cached constant memory
    PARAM = "param"        # Kernel invocation parameter buffer


class BaseType:
    """Base class for all GPU IR types."""
    pass


@dataclass(frozen=True)
class PrimitiveType(BaseType):
    """Primitive scalar GPU types."""
    name: str
    bitwidth: int

    def __str__(self) -> str:
        return self.name

    @property
    def is_float(self) -> bool:
        return self.name in {"f16", "f32", "f64", "bf16"}

    @property
    def is_int(self) -> bool:
        return self.name in {"s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64"}

    @property
    def is_predicate(self) -> bool:
        return self.name == "pred"


# Standard Type Singletons
F16 = PrimitiveType("f16", 16)
F32 = PrimitiveType("f32", 32)
F64 = PrimitiveType("f64", 64)
BF16 = PrimitiveType("bf16", 16)

S8 = PrimitiveType("s8", 8)
S16 = PrimitiveType("s16", 16)
S32 = PrimitiveType("s32", 32)
S64 = PrimitiveType("s64", 64)

U8 = PrimitiveType("u8", 8)
U16 = PrimitiveType("u16", 16)
U32 = PrimitiveType("u32", 32)
U64 = PrimitiveType("u64", 64)

B8 = PrimitiveType("b8", 8)
B16 = PrimitiveType("b16", 16)
B32 = PrimitiveType("b32", 32)
B64 = PrimitiveType("b64", 64)

PRED = PrimitiveType("pred", 1)


@dataclass(frozen=True)
class PointerType(BaseType):
    """Typed pointer targeting a specific GPU address space."""
    pointee: BaseType
    space: AddressSpace = AddressSpace.GLOBAL

    def __str__(self) -> str:
        return f"{self.space.value}_ptr<{self.pointee}>"


@dataclass(frozen=True)
class VectorType(BaseType):
    """Vector type (e.g. float4, uvec2)."""
    element_type: PrimitiveType
    count: int

    def __str__(self) -> str:
        return f"{self.element_type}x{self.count}"


@dataclass(frozen=True)
class CooperativeMatrixType(BaseType):
    """Tile representation for Tensor / Matrix Core acceleration."""
    element_type: PrimitiveType
    rows: int
    cols: int
    use: str  # 'matrix_a', 'matrix_b', 'accumulator'

    def __str__(self) -> str:
        return f"coop_matrix<{self.element_type}, {self.rows}x{self.cols}, {self.use}>"


def parse_type(type_str: str) -> BaseType:
    """Parse PTX type identifier into IR BaseType."""
    cleaned = type_str.lstrip(".").lower()
    mapping = {
        "f16": F16, "f32": F32, "f64": F64, "bf16": BF16,
        "s8": S8, "s16": S16, "s32": S32, "s64": S64,
        "u8": U8, "u16": U16, "u32": U32, "u64": U64,
        "b8": B8, "b16": B16, "b32": B32, "b64": B64,
        "pred": PRED,
    }
    if cleaned in mapping:
        return mapping[cleaned]
    # Default fallback to b32 if unknown
    return B32
