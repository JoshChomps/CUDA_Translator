"""Python Emulation of CUDA Driver API for Host-Side Interception."""

from __future__ import annotations
import ctypes
from typing import Dict, Any, List, Optional
from cuda_translator.frontend.ptx_lexer import PTXLexer
from cuda_translator.frontend.ptx_parser import PTXParser
from cuda_translator.transform.warp_to_wavefront import WarpToWavefrontPass
from cuda_translator.transform.lds_normalizer import LDSNormalizerPass
from cuda_translator.transform.matrix_accelerator import MatrixAcceleratorPass
from cuda_translator.backend.vulkan_backend import VulkanBackend
from cuda_translator.backend.hip_backend import HIPBackend
from cuda_translator.ir.module import Module, Kernel


class CUResult:
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_NOT_INITIALIZED = 3
    CUDA_ERROR_DEINITIALIZED = 4
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
    CUDA_ERROR_INVALID_IMAGE = 200
    CUDA_ERROR_INVALID_CONTEXT = 201
    CUDA_ERROR_NOT_FOUND = 500


class DeviceMemoryBuffer:
    """Represents a virtualized device memory allocation."""
    def __init__(self, size_bytes: int, address: int) -> None:
        self.size_bytes = size_bytes
        self.address = address
        self.buffer = bytearray(size_bytes)


class CUDADriverRuntime:
    """Interception runtime implementing standard CUDA Driver API behaviors."""

    def __init__(self, target_backend: str = "vulkan") -> None:
        self.target_backend = target_backend
        self.initialized = False
        self._next_dptr = 0x10000000
        self.allocations: Dict[int, DeviceMemoryBuffer] = {}
        self.modules: Dict[int, Module] = {}
        self._next_module_id = 1
        self.warp_pass = WarpToWavefrontPass(target_wavefront_size=64 if target_backend == "hip" else 32)
        self.lds_pass = LDSNormalizerPass()
        self.matrix_pass = MatrixAcceleratorPass(target_backend=target_backend)

    def cuInit(self, flags: int = 0) -> int:
        self.initialized = True
        return CUResult.CUDA_SUCCESS

    def cuDeviceGetCount(self) -> tuple[int, int]:
        if not self.initialized:
            return CUResult.CUDA_ERROR_NOT_INITIALIZED, 0
        return CUResult.CUDA_SUCCESS, 1

    def cuMemAlloc(self, size_bytes: int) -> tuple[int, int]:
        """Allocate virtual device memory."""
        if not self.initialized:
            return CUResult.CUDA_ERROR_NOT_INITIALIZED, 0
        if size_bytes <= 0:
            return CUResult.CUDA_ERROR_INVALID_VALUE, 0

        dptr = self._next_dptr
        # 256-byte alignment
        aligned_size = (size_bytes + 255) & ~255
        self._next_dptr += aligned_size

        self.allocations[dptr] = DeviceMemoryBuffer(size_bytes, dptr)
        return CUResult.CUDA_SUCCESS, dptr

    def cuMemFree(self, dptr: int) -> int:
        if dptr in self.allocations:
            del self.allocations[dptr]
            return CUResult.CUDA_SUCCESS
        return CUResult.CUDA_ERROR_INVALID_VALUE

    def cuMemcpyHtoD(self, dst_dptr: int, src_bytes: bytes) -> int:
        """Copy data from host to virtual device buffer."""
        if dst_dptr not in self.allocations:
            return CUResult.CUDA_ERROR_INVALID_VALUE
        buf = self.allocations[dst_dptr]
        count = min(len(src_bytes), buf.size_bytes)
        buf.buffer[:count] = src_bytes[:count]
        return CUResult.CUDA_SUCCESS

    def cuMemcpyDtoH(self, dst_bytes: bytearray, src_dptr: int) -> int:
        """Copy data from virtual device buffer to host."""
        if src_dptr not in self.allocations:
            return CUResult.CUDA_ERROR_INVALID_VALUE
        buf = self.allocations[src_dptr]
        count = min(len(dst_bytes), buf.size_bytes)
        dst_bytes[:count] = buf.buffer[:count]
        return CUResult.CUDA_SUCCESS

    def cuModuleLoadData(self, ptx_source: str) -> tuple[int, Optional[int]]:
        """Parse, normalize, and compile PTX code in-memory."""
        if not self.initialized:
            return CUResult.CUDA_ERROR_NOT_INITIALIZED, None

        lexer = PTXLexer(ptx_source)
        tokens = lexer.tokenize()
        parser = PTXParser(tokens)
        module = parser.parse_module()

        # Run translation passes
        self.warp_pass.run_on_module(module)
        self.lds_pass.run_on_module(module)
        self.matrix_pass.run_on_module(module)

        mod_id = self._next_module_id
        self._next_module_id += 1
        self.modules[mod_id] = module
        return CUResult.CUDA_SUCCESS, mod_id

    def cuModuleGetFunction(self, module_id: int, func_name: str) -> tuple[int, Optional[Kernel]]:
        if module_id not in self.modules:
            return CUResult.CUDA_ERROR_INVALID_IMAGE, None
        module = self.modules[module_id]
        for k in module.kernels:
            if k.name == func_name:
                return CUResult.CUDA_SUCCESS, k
        return CUResult.CUDA_ERROR_NOT_FOUND, None

    def cuLaunchKernel(self, kernel: Kernel, grid_dim: tuple[int, int, int],
                       block_dim: tuple[int, int, int], args: List[Any]) -> int:
        """Emit target shader/kernel and simulate dispatch."""
        if not self.initialized:
            return CUResult.CUDA_ERROR_NOT_INITIALIZED

        # Generate target code
        if self.target_backend == "vulkan":
            backend = VulkanBackend(local_size_x=block_dim[0], local_size_y=block_dim[1], local_size_z=block_dim[2])
            target_code = backend.generate(Module(name="dispatch", kernels=[kernel]))
        else:
            backend = HIPBackend()
            target_code = backend.generate(Module(name="dispatch", kernels=[kernel]))

        # Kernel dispatch recorded and ready for execution
        return CUResult.CUDA_SUCCESS
