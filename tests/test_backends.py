"""Unit tests for Vulkan and AMD HIP backend emitters."""

import pytest
from cuda_translator.frontend.ptx_lexer import PTXLexer
from cuda_translator.frontend.ptx_parser import PTXParser
from cuda_translator.backend.vulkan_backend import VulkanBackend
from cuda_translator.backend.hip_backend import HIPBackend

PTX_SOURCE = """
.version 7.0
.target sm_80
.visible .entry sample_kernel(.param .u64 in_ptr, .param .u64 out_ptr)
{
    .reg .b32 %r<5>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %tid.x;
    ld.global.f32 %f0, [%rd0];
    bar.sync 0;
    shfl.sync.bfly.b32 %f1, %f0, 16, 0x1f, 0xffffffff;
    st.global.f32 [%rd1], %f1;
    ret;
}
"""


def test_vulkan_backend_emission():
    lexer = PTXLexer(PTX_SOURCE)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    backend = VulkanBackend()
    glsl = backend.generate(module)

    assert "#version 450" in glsl
    assert "GL_KHR_shader_subgroup_shuffle" in glsl
    assert "layout(local_size_x = 32" in glsl
    assert "subgroupShuffleXor" in glsl
    assert "barrier();" in glsl


def test_hip_backend_emission():
    lexer = PTXLexer(PTX_SOURCE)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    backend = HIPBackend()
    hip_cpp = backend.generate(module)

    assert "#include <hip/hip_runtime.h>" in hip_cpp
    assert "__global__ void sample_kernel" in hip_cpp
    assert "__syncthreads();" in hip_cpp
    assert "__shfl_xor" in hip_cpp
