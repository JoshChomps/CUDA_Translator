"""Matrix Acceleration Routing Pass.

Abstracts vendor-specific matrix multiplication primitives, recognizing
CUDA WMMA Tensor Core instructions and routing them to target hardware units
(such as AMD CDNA MFMA, AMD RDNA WMMA, or Vulkan Cooperative Matrix).
"""

from __future__ import annotations
from cuda_translator.ir.module import Module, Kernel
from cuda_translator.ir.instructions import (
    CooperativeMatrixLoadInst, CooperativeMatrixStoreInst, CooperativeMatrixMulAddInst
)
from cuda_translator.ir.types import CooperativeMatrixType, F16, F32


class MatrixAcceleratorPass:
    """Pass that routes WMMA Tensor Core operations to target matrix hardware."""

    def __init__(self, target_backend: str = "vulkan", tile_m: int = 16, tile_n: int = 16, tile_k: int = 16) -> None:
        self.target_backend = target_backend
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

    def run_on_module(self, module: Module) -> Module:
        for kernel in module.kernels:
            self.run_on_kernel(kernel)
        return module

    def run_on_kernel(self, kernel: Kernel) -> None:
        """Analyze and bind cooperative matrix types for fragments in kernel."""
        for block in kernel.blocks:
            for inst in block.instructions:
                if isinstance(inst, CooperativeMatrixLoadInst):
                    # Bind fragment type
                    kernel.registers[inst.dest_frag] = CooperativeMatrixType(
                        element_type=F16,
                        rows=self.tile_m,
                        cols=self.tile_k,
                        use="matrix_a"
                    )
                elif isinstance(inst, CooperativeMatrixMulAddInst):
                    # Accumulator fragment
                    kernel.registers[inst.dest_frag] = CooperativeMatrixType(
                        element_type=F32,
                        rows=self.tile_m,
                        cols=self.tile_n,
                        use="accumulator"
                    )
