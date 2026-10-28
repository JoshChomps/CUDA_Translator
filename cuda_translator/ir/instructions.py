"""Instruction set definitions for the Neutral GPU Intermediate Representation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any
from cuda_translator.ir.types import BaseType, AddressSpace, B32, PRED


@dataclass
class Instruction:
    """Base class for all neutral GPU IR instructions."""
    predicate: Optional[str] = None
    pred_negated: bool = False

    def is_predicated(self) -> bool:
        return self.predicate is not None


@dataclass
class AllocaInst(Instruction):
    """Memory allocation in Shared (LDS) or Local (Stack) address space."""
    name: str = ""
    element_type: BaseType = B32
    count: int = 1
    space: AddressSpace = AddressSpace.SHARED


@dataclass
class LoadInst(Instruction):
    """Load from memory into a virtual register."""
    dest: str = ""
    ptr: str = ""
    data_type: BaseType = B32
    space: AddressSpace = AddressSpace.GLOBAL
    offset: int = 0


@dataclass
class StoreInst(Instruction):
    """Store from a virtual register into memory."""
    ptr: str = ""
    src: str = ""
    data_type: BaseType = B32
    space: AddressSpace = AddressSpace.GLOBAL
    offset: int = 0


@dataclass
class BinaryOpInst(Instruction):
    """Binary arithmetic, logical, or bitwise operation."""
    dest: str = ""
    op: str = "add"  # add, sub, mul, div, rem, and, or, xor, shl, shr
    src1: str = ""
    src2: str = ""
    data_type: BaseType = B32


@dataclass
class UnaryOpInst(Instruction):
    """Unary operation (negation, absolute value, sqrt)."""
    dest: str = ""
    op: str = "neg"  # neg, not, abs, sqrt, rsqrt
    src: str = ""
    data_type: BaseType = B32


@dataclass
class FusedMultiplyAddInst(Instruction):
    """Fused multiply-add: dest = (a * b) + c."""
    dest: str = ""
    src_a: str = ""
    src_b: str = ""
    src_c: str = ""
    data_type: BaseType = B32


@dataclass
class CompareInst(Instruction):
    """Comparison generating a 1-bit predicate."""
    dest: str = ""  # predicate register (e.g., %p1)
    cond: str = "eq"  # eq, ne, lt, le, gt, ge
    src1: str = ""
    src2: str = ""
    data_type: BaseType = B32


@dataclass
class ConvertInst(Instruction):
    """Type conversion / casting between bitwidths and formats."""
    dest: str = ""
    src: str = ""
    from_type: BaseType = B32
    to_type: BaseType = B32


@dataclass
class BranchInst(Instruction):
    """Control flow branch (conditional or unconditional)."""
    target: str = ""  # BasicBlock label
    condition: Optional[str] = None


@dataclass
class BarrierInst(Instruction):
    """Workgroup / block synchronization barrier."""
    kind: str = "workgroup"  # workgroup, memory_fence


@dataclass
class SubgroupShuffleInst(Instruction):
    """Intra-warp / intra-subgroup register exchange intrinsic."""
    dest: str = ""
    src: str = ""
    delta_or_lane: str = ""
    width: int = 32
    mode: str = "bfly"  # bfly (butterfly/xor), down, up, idx


@dataclass
class SubgroupBallotInst(Instruction):
    """Subgroup active thread ballot."""
    dest: str = ""
    pred: str = ""
    width: int = 32


@dataclass
class CooperativeMatrixLoadInst(Instruction):
    """Load matrix tile into cooperative matrix fragment."""
    dest_frag: str = ""
    ptr: str = ""
    stride: str = ""
    layout: str = "row_major"  # row_major, col_major


@dataclass
class CooperativeMatrixStoreInst(Instruction):
    """Store cooperative matrix fragment into memory."""
    ptr: str = ""
    src_frag: str = ""
    stride: str = ""
    layout: str = "row_major"


@dataclass
class CooperativeMatrixMulAddInst(Instruction):
    """Matrix Multiply-Accumulate: D = A * B + C."""
    dest_frag: str = ""
    frag_a: str = ""
    frag_b: str = ""
    frag_c: str = ""


@dataclass
class ReturnInst(Instruction):
    """Return from kernel execution."""
    pass
