"""Warp-to-Wavefront Scheduling & Microarchitectural Normalization Pass.

Maps NVIDIA 32-thread warp semantics onto target AMD 64-thread wavefronts
or universal Vulkan subgroup architectures without performance degradation
or synchronization deadlocks.
"""

from __future__ import annotations
from typing import Optional
from cuda_translator.ir.module import Module, Kernel, BasicBlock
from cuda_translator.ir.instructions import (
    Instruction, SubgroupShuffleInst, SubgroupBallotInst, BinaryOpInst, CompareInst, ConvertInst
)
from cuda_translator.ir.types import U32, U64, B32, B64, PRED


class WarpToWavefrontPass:
    """Transforms 32-lane warp primitives to 64-lane wavefront or Vulkan subgroup ops."""

    def __init__(self, target_wavefront_size: int = 64, enable_dual_warp_coalescing: bool = False) -> None:
        self.target_wavefront_size = target_wavefront_size
        self.enable_dual_warp_coalescing = enable_dual_warp_coalescing

    def run_on_module(self, module: Module) -> Module:
        for kernel in module.kernels:
            self.run_on_kernel(kernel)
        return module

    def run_on_kernel(self, kernel: Kernel) -> None:
        """Process all basic blocks within a compute kernel."""
        for block in kernel.blocks:
            new_instructions = []
            for inst in block.instructions:
                transformed = self.transform_instruction(inst, kernel)
                if isinstance(transformed, list):
                    new_instructions.extend(transformed)
                else:
                    new_instructions.append(transformed)
            block.instructions = new_instructions

    def transform_instruction(self, inst: Instruction, kernel: Kernel) -> Instruction | list[Instruction]:
        """Normalize warp intrinsics for wavefront microarchitectures."""

        # 1. Transform Intra-Warp Shuffle (__shfl_sync)
        if isinstance(inst, SubgroupShuffleInst):
            if self.target_wavefront_size == 64:
                # In 64-lane wavefront, an intra-warp shuffle must prevent cross-warp lane bleeding.
                # A 32-lane warp only exchanges data within lanes [0..31] or [32..63].
                # We update the instruction's width metadata to 32 to tell backends
                # to emit sub-wavefront clustered shuffles (e.g. subgroupClusteredShuffle or ds_swizzle)
                inst.width = 32
                return inst
            elif self.target_wavefront_size == 32:
                # Direct 1:1 hardware match (e.g. AMD RDNA in wave32 mode)
                inst.width = 32
                return inst

        # 2. Transform Warp Ballot (__ballot_sync)
        elif isinstance(inst, SubgroupBallotInst):
            if self.target_wavefront_size == 64:
                # NVIDIA __ballot_sync returns a 32-bit activemask for 32 threads.
                # In AMD wave64, EXEC register is 64 bits.
                # We convert the destination register type to 64-bit or extract the 32-bit sub-mask.
                inst.width = 64
                kernel.registers[inst.dest] = B64
                return inst

        return inst
