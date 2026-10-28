"""Module and Kernel structures for the Neutral GPU Intermediate Representation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from cuda_translator.ir.types import BaseType, AddressSpace, B32
from cuda_translator.ir.instructions import Instruction, AllocaInst


@dataclass
class KernelParam:
    """Kernel launch argument parameter."""
    name: str
    data_type: BaseType
    is_pointer: bool = False
    space: AddressSpace = AddressSpace.GLOBAL

    def __str__(self) -> str:
        ptr_suffix = "*" if self.is_pointer else ""
        return f"{self.data_type}{ptr_suffix} {self.name}"


@dataclass
class BasicBlock:
    """Basic block of linear instructions terminated by a branch or return."""
    label: str
    instructions: List[Instruction] = field(default_factory=list)

    def add_instruction(self, inst: Instruction) -> None:
        self.instructions.append(inst)


@dataclass
class Kernel:
    """GPU Compute Kernel function representation."""
    name: str
    params: List[KernelParam] = field(default_factory=list)
    blocks: List[BasicBlock] = field(default_factory=list)
    shared_allocs: List[AllocaInst] = field(default_factory=list)
    registers: Dict[str, BaseType] = field(default_factory=dict)
    workgroup_size: Optional[tuple[int, int, int]] = None

    def get_or_create_block(self, label: str) -> BasicBlock:
        for block in self.blocks:
            if block.label == label:
                return block
        new_block = BasicBlock(label=label)
        self.blocks.append(new_block)
        return new_block


@dataclass
class Module:
    """Translation unit containing GPU kernels and global definitions."""
    name: str
    kernels: List[Kernel] = field(default_factory=list)

    def add_kernel(self, kernel: Kernel) -> None:
        self.kernels.append(kernel)
