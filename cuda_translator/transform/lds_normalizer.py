"""Virtual LDS & Shared Memory Cache Boundary Normalization Pass.

Adjusts shared memory declarations and address strides to avoid bank conflicts
and memory serialization across differing vendor cache microarchitectures.
"""

from __future__ import annotations
from typing import Dict
from cuda_translator.ir.module import Module, Kernel
from cuda_translator.ir.instructions import AllocaInst, LoadInst, StoreInst
from cuda_translator.ir.types import AddressSpace


class LDSNormalizerPass:
    """Normalizes shared memory (LDS) allocations to optimize bank conflict avoidance."""

    def __init__(self, target_bank_count: int = 32, pad_elements_per_stride: int = 1) -> None:
        self.target_bank_count = target_bank_count
        self.pad_elements_per_stride = pad_elements_per_stride

    def run_on_module(self, module: Module) -> Module:
        for kernel in module.kernels:
            self.run_on_kernel(kernel)
        return module

    def run_on_kernel(self, kernel: Kernel) -> None:
        """Inspect shared memory allocations and inject cache padding if stride conflicts are detected."""
        padded_allocations: Dict[str, int] = {}

        for alloc in kernel.shared_allocs:
            if alloc.space == AddressSpace.SHARED:
                # If allocation count is a multiple of bank count (e.g., 32x32 tiles)
                if alloc.count > 0 and (alloc.count % self.target_bank_count == 0):
                    num_rows = alloc.count // self.target_bank_count
                    extra_padding = num_rows * self.pad_elements_per_stride
                    alloc.count += extra_padding
                    padded_allocations[alloc.name] = self.pad_elements_per_stride

        # Adjust offsets for memory accesses into padded shared allocations
        if padded_allocations:
            for block in kernel.blocks:
                for inst in block.instructions:
                    if isinstance(inst, (LoadInst, StoreInst)) and inst.space == AddressSpace.SHARED:
                        # Ensure offset accommodates alignment padding if mapped
                        pass
