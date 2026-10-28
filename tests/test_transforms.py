"""Unit tests for Microarchitectural Transformation Passes."""

import pytest
from cuda_translator.ir.module import Module, Kernel, BasicBlock
from cuda_translator.ir.instructions import (
    SubgroupShuffleInst, SubgroupBallotInst, AllocaInst, CooperativeMatrixLoadInst,
    CooperativeMatrixMulAddInst
)
from cuda_translator.ir.types import B32, B64, F32, AddressSpace
from cuda_translator.transform.warp_to_wavefront import WarpToWavefrontPass
from cuda_translator.transform.lds_normalizer import LDSNormalizerPass
from cuda_translator.transform.matrix_accelerator import MatrixAcceleratorPass


def test_warp_to_wavefront_pass_wave64():
    kernel = Kernel(name="test_warp")
    bb = kernel.get_or_create_block("entry")
    bb.add_instruction(SubgroupShuffleInst(dest="%r1", src="%r0", delta_or_lane="16", width=32))
    bb.add_instruction(SubgroupBallotInst(dest="%r2", pred="%p0", width=32))

    module = Module(name="test_mod", kernels=[kernel])
    pass_wave64 = WarpToWavefrontPass(target_wavefront_size=64)
    pass_wave64.run_on_module(module)

    insts = kernel.blocks[0].instructions
    shuffle_inst = insts[0]
    ballot_inst = insts[1]

    assert isinstance(shuffle_inst, SubgroupShuffleInst)
    assert shuffle_inst.width == 32  # Sub-warp clamped
    assert isinstance(ballot_inst, SubgroupBallotInst)
    assert ballot_inst.width == 64
    assert kernel.registers["%r2"] == B64


def test_lds_normalizer_padding():
    kernel = Kernel(name="test_lds")
    # Allocate a 32x32 = 1024 element shared buffer (prone to 32-bank conflict)
    kernel.shared_allocs.append(
        AllocaInst(name="smem", element_type=F32, count=1024, space=AddressSpace.SHARED)
    )

    module = Module(name="test_mod", kernels=[kernel])
    pass_lds = LDSNormalizerPass(target_bank_count=32, pad_elements_per_stride=1)
    pass_lds.run_on_module(module)

    # 1024 elements = 32 rows of 32 elements. Padded with 1 element per row -> 1024 + 32 = 1056 elements
    assert kernel.shared_allocs[0].count == 1056


def test_matrix_accelerator_pass():
    kernel = Kernel(name="test_wmma")
    bb = kernel.get_or_create_block("entry")
    bb.add_instruction(CooperativeMatrixLoadInst(dest_frag="%rd4", ptr="%rd0", stride="16"))
    bb.add_instruction(CooperativeMatrixMulAddInst(dest_frag="%rd7", frag_a="%rd4", frag_b="%rd5", frag_c="%rd6"))

    module = Module(name="test_mod", kernels=[kernel])
    pass_matrix = MatrixAcceleratorPass(target_backend="vulkan", tile_m=16, tile_n=16, tile_k=16)
    pass_matrix.run_on_module(module)

    assert "%rd4" in kernel.registers
    assert "%rd7" in kernel.registers
    assert kernel.registers["%rd4"].use == "matrix_a"
    assert kernel.registers["%rd7"].use == "accumulator"
