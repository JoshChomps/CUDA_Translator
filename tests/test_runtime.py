"""Unit tests for CUDA Driver API runtime emulation."""

import pytest
from cuda_translator.runtime.driver_api import CUDADriverRuntime, CUResult

SAMPLE_PTX = """
.version 7.0
.target sm_80
.visible .entry dummy_kernel(.param .u64 ptr)
{
    .reg .b32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""


def test_driver_runtime_lifecycle():
    runtime = CUDADriverRuntime(target_backend="vulkan")

    # Before init
    res, count = runtime.cuDeviceGetCount()
    assert res == CUResult.CUDA_ERROR_NOT_INITIALIZED

    # Init
    assert runtime.cuInit() == CUResult.CUDA_SUCCESS
    res, count = runtime.cuDeviceGetCount()
    assert res == CUResult.CUDA_SUCCESS
    assert count == 1

    # Alloc
    res, dptr = runtime.cuMemAlloc(1024)
    assert res == CUResult.CUDA_SUCCESS
    assert dptr > 0

    # Memcpy HtoD
    host_data = b"HelloGPU" * 8
    res = runtime.cuMemcpyHtoD(dptr, host_data)
    assert res == CUResult.CUDA_SUCCESS

    # Memcpy DtoH
    readback = bytearray(len(host_data))
    res = runtime.cuMemcpyDtoH(readback, dptr)
    assert res == CUResult.CUDA_SUCCESS
    assert bytes(readback) == host_data

    # Load Module
    res, mod_id = runtime.cuModuleLoadData(SAMPLE_PTX)
    assert res == CUResult.CUDA_SUCCESS
    assert mod_id is not None

    # Get Function
    res, func = runtime.cuModuleGetFunction(mod_id, "dummy_kernel")
    assert res == CUResult.CUDA_SUCCESS
    assert func is not None

    # Launch Kernel
    res = runtime.cuLaunchKernel(func, (1, 1, 1), (32, 1, 1), [dptr])
    assert res == CUResult.CUDA_SUCCESS

    # Free
    assert runtime.cuMemFree(dptr) == CUResult.CUDA_SUCCESS
