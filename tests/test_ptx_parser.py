"""Unit tests for PTX Lexer and Parser."""

import pytest
from cuda_translator.frontend.ptx_lexer import PTXLexer, TokenType
from cuda_translator.frontend.ptx_parser import PTXParser
from cuda_translator.ir.types import F32, B32, AddressSpace
from cuda_translator.ir.instructions import (
    LoadInst, StoreInst, BinaryOpInst, SubgroupShuffleInst, CooperativeMatrixMulAddInst
)


SAMPLE_VECTOR_ADD = """
.version 7.0
.target sm_80
.visible .entry vec_add(.param .u64 a_ptr, .param .u64 b_ptr, .param .f32 scalar)
{
    .reg .b32 %r<5>;
    .reg .f32 %f<3>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.global.f32 %f0, [%rd0];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd2], %f2;
    ret;
}
"""


def test_lexer_tokenization():
    lexer = PTXLexer(SAMPLE_VECTOR_ADD)
    tokens = lexer.tokenize()
    assert len(tokens) > 10
    directives = [t.value for t in tokens if t.type == TokenType.DIRECTIVE]
    assert ".version" in directives
    assert ".entry" in directives
    assert ".reg" in directives


def test_parser_kernel_structure():
    lexer = PTXLexer(SAMPLE_VECTOR_ADD)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    assert len(module.kernels) == 1
    kernel = module.kernels[0]
    assert kernel.name == "vec_add"
    assert len(kernel.params) == 3
    assert kernel.params[0].is_pointer is True
    assert kernel.params[2].is_pointer is False


def test_parser_instructions():
    lexer = PTXLexer(SAMPLE_VECTOR_ADD)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    kernel = module.kernels[0]
    insts = [inst for block in kernel.blocks for inst in block.instructions]

    load_insts = [i for i in insts if isinstance(i, LoadInst)]
    add_insts = [i for i in insts if isinstance(i, BinaryOpInst)]
    store_insts = [i for i in insts if isinstance(i, StoreInst)]

    assert len(load_insts) == 1
    assert load_insts[0].space == AddressSpace.GLOBAL
    assert len(add_insts) == 1
    assert add_insts[0].op == "add"
    assert len(store_insts) == 1
