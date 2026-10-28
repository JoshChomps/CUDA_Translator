"""CUDA_Translator: Cross-Architecture GPU Translation Layer.

Translates NVIDIA CUDA PTX virtual ISA and Driver API invocations into
neutral GPU IR, applying hardware normalization passes to target AMD ROCm/HIP
and universal Vulkan Compute (SPIR-V).
"""

__version__ = "0.1.0"
