"""CUDA Runtime and Driver API emulation and interception."""

from cuda_translator.runtime.driver_api import CUDADriverRuntime, CUResult

__all__ = ["CUDADriverRuntime", "CUResult"]
