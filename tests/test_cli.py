"""Integration tests for the CUDA_Translator CLI and benchmark kernels."""

from pathlib import Path
import pytest
from cuda_translator.frontend.ptx_lexer import PTXLexer
from cuda_translator.frontend.ptx_parser import PTXParser
from cuda_translator.transform.warp_to_wavefront import WarpToWavefrontPass
from cuda_translator.transform.lds_normalizer import LDSNormalizerPass
from cuda_translator.transform.matrix_accelerator import MatrixAcceleratorPass
from cuda_translator.backend.vulkan_backend import VulkanBackend
from cuda_translator.backend.hip_backend import HIPBackend


BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks" / "kernels"


@pytest.mark.parametrize("kernel_file", ["vector_add.ptx", "warp_shuffle.ptx", "matrix_wmma.ptx"])
def test_translate_benchmark_to_vulkan(kernel_file: str):
    file_path = BENCHMARKS_DIR / kernel_file
    assert file_path.exists(), f"Benchmark kernel file {file_path} missing"

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lexer = PTXLexer(source)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    # Apply transformation passes
    WarpToWavefrontPass(target_wavefront_size=32).run_on_module(module)
    LDSNormalizerPass().run_on_module(module)
    MatrixAcceleratorPass(target_backend="vulkan").run_on_module(module)

    # Emit Vulkan
    backend = VulkanBackend()
    code = backend.generate(module)
    assert len(code) > 100
    assert "#version 450" in code


@pytest.mark.parametrize("kernel_file", ["vector_add.ptx", "warp_shuffle.ptx", "matrix_wmma.ptx"])
def test_translate_benchmark_to_hip(kernel_file: str):
    file_path = BENCHMARKS_DIR / kernel_file
    assert file_path.exists(), f"Benchmark kernel file {file_path} missing"

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lexer = PTXLexer(source)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module()

    # Apply transformation passes
    WarpToWavefrontPass(target_wavefront_size=64).run_on_module(module)
    LDSNormalizerPass().run_on_module(module)
    MatrixAcceleratorPass(target_backend="hip").run_on_module(module)

    # Emit HIP
    backend = HIPBackend(target_wavefront_size=64)
    code = backend.generate(module)
    assert len(code) > 100
    assert "#include <hip/hip_runtime.h>" in code
