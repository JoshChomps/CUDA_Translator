# GPU Driver Architecture, Kernel Space vs. User Space, and CUDA Runtime Interception

This document provides a comprehensive systems-level specification of the modern GPU computing stack, detailing physical hardware initialization, OS Kernel Drivers (Ring 0), User-Space Driver Runtimes (Ring 3), and practical translation layer interception strategies for **CUDA_Translator**.

---

## 1. The Full-Stack GPU Hierarchy

When an application executes a GPU compute kernel (e.g. `matrix_multiply<<<grid, block>>>(A, B, C)`), the request traverses six distinct layers of hardware and software abstraction:

```
+-----------------------------------------------------------------------------------------------+
| LAYER 6: USER APPLICATION      | PyTorch, JAX, llama.cpp, Kinesis, Custom Neural Models        |
+-----------------------------------------------------------------------------------------------+
| LAYER 5: HIGH-LEVEL FRAMEWORKS | CUDA Runtime API (libcudart.so / cudart64.dll)                |
|                                | Memory wrappers, stream management, event tracking           |
+-----------------------------------------------------------------------------------------------+
| LAYER 4: CUDA DRIVER API       | User-Space Driver API (libcuda.so / nvcuda.dll)               |
|                                | cuLaunchKernel, cuMemAlloc, cuCtxCreate, PTX JIT compilation  |
+-----------------------------------------------------------------------------------------------+
| LAYER 3: TRANSLATION / COMPUTE | >>> CUDA_TRANSLATOR / ZLUDA / ChipStar / Vulkan / HIP <<<    |
|                                | Dynamic API interception, PTX-to-SPIR-V/ROCm JIT translation  |
+-----------------------------------------------------------------------------------------------+
| LAYER 2: OS KERNEL DRIVER      | Ring 0 Kernel Modules:                                       |
|                                | - Windows: WDDM Driver (nvlddmkm.sys / amdkmdag.sys)         |
|                                | - Linux: Direct Rendering Manager (nvidia.ko / amdgpu.ko)    |
|                                | Page table manipulation, interrupt handling, PCIe DMA rings   |
+-----------------------------------------------------------------------------------------------+
| LAYER 1: GPU FIRMWARE & ASIC   | Microcontroller System Processor (GSP / PM4 / PSP),          |
|                                | Command Processors, SMs/Compute Units, HBM/GDDR6X VRAM       |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Kernel Space (Ring 0) vs. User Space (Ring 3): Why Translation Lives in Ring 3

A fundamental architectural question in GPU systems engineering is whether a hardware compatibility or translation layer should be implemented as an **OS Device Driver (Ring 0)** or a **User-Space Runtime (Ring 3)**.

### The Anatomy of an OS Device Driver (Ring 0)
The kernel driver (`nvlddmkm.sys` on Windows, `nvidia.ko` on Linux) possesses raw supervisor privileges and is responsible for:
1. **PCIe Base Address Register (BAR) Mapping**: Mapping the physical electrical registers of the GPU into the CPU's physical address space.
2. **Interrupt Service Routines (ISRs)**: Handling hardware interrupts triggered when a compute command completes, a page fault occurs, or thermal thresholds are tripped.
3. **Firmware Loading**: Bootstrapping the on-chip GPU System Processor (GSP) microcode.
4. **Physical Memory Management**: Managing hardware IOMMU page tables, allocating contiguous physical video memory (VRAM), and establishing DMA rings.
5. **Hardware Submission Queues**: Writing command buffer pointers directly into physical GPU doorbell registers over the PCIe bus.

### Why Writing a Kernel Driver for CUDA Translation is Fatal
* **Kernel Panics & BSODs**: Any minor bug, memory boundary overflow, or unhandled pointer in Ring 0 crashes the entire operating system immediately (Blue Screen of Death on Windows, Kernel Panic on Linux).
* **Cryptographic Code Signing**: Modern operating systems strictly reject unsigned kernel drivers. Windows requires Microsoft WHQL (Windows Hardware Quality Labs) EV Code Signing certification, while Linux distributions with Secure Boot reject unsigned `.ko` modules.
* **Privilege Requirements**: A kernel driver requires administrative/root privileges to install and update, making distribution impossible in sandboxed environments, cloud clusters, or standard user installations.
* **Architectural Inefficiency**: Translating compute kernels does not require manipulating PCIe registers. CUDA, Vulkan, Metal, and ROCm all expose complete, high-performance **User-Space APIs (Ring 3)** that provide direct command stream recording with zero kernel context switches.

> [!IMPORTANT]
> **Architectural Rule**: CUDA translation layers (including `CUDA_Translator`, ZLUDA, and chipStar) operate strictly in **User Space (Ring 3)**. They achieve bare-metal GPU performance by intercepting User-Space Driver APIs and routing translated command streams to standard vendor runtimes (Vulkan, ROCm/HIP, or Metal).

---

## 3. The Two CUDA APIs: Driver API vs. Runtime API

Understanding the division between NVIDIA's user-space libraries is essential for clean translation interception:

```
                  +-----------------------------------+
                  |      Application / PyTorch        |
                  +-----------------------------------+
                       │                         │
      (High-Level C++ Syntax)       (Explicit Low-Level Handle)
                       │                         │
                       ▼                         │
          +-------------------------+            │
          |     CUDA Runtime API    |            │
          |      (libcudart.so)     |            │
          |  - cudaMalloc           |            │
          |  - cudaMemcpy           |            │
          |  - cudaLaunchKernel     |            │
          +-------------------------+            │
                       │                         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                      +---------------------------+
                      |      CUDA Driver API      |
                      |       (libcuda.so)        |
                      |  - cuMemAlloc             |
                      |  - cuLaunchKernel         |
                      |  - cuModuleLoadData (PTX) |
                      +---------------------------+
                                    │
                                    ▼
                      +---------------------------+
                      |     OS Kernel Driver      |
                      |   (nvidia.ko / ioctl)     |
                      +---------------------------+
```

### 1. CUDA Runtime API (`libcudart.so` / `cudart64.dll`)
* Designed for ergonomics and compiler integration.
* Automatically initializes the GPU context, manages primary context lifecycles, and translates high-level kernel launch syntax (`kernel<<<grid, block>>>()`) into internal driver calls.
* Applications can statically link `libcudart.a` or dynamically load it.

### 2. CUDA Driver API (`libcuda.so` / `nvcuda.dll`)
* The fundamental low-level user-space driver interface installed with the NVIDIA display driver.
* Deals with explicit handles: `CUdevice`, `CUcontext`, `CUmodule`, `CUfunction`, `CUstream`.
* Accepts raw PTX text or cubin binaries at runtime (`cuModuleLoadData`), invokes the internal JIT compiler, and dispatches parameter structures directly into hardware streams via `cuLaunchKernel`.

---

## 4. CUDA_Translator Interception Architecture

`CUDA_Translator` enables non-NVIDIA GPUs (AMD, Intel, Apple Silicon, and ARM Mali) to execute CUDA applications through user-space API interception and just-in-time shader translation.

### Interception Methods

```
+-----------------------------------------------------------------------------------------------+
|                                INTERCEPTION METHODOLOGIES                                     |
|                                                                                               |
|  1. SHARED LIBRARY REPLACEMENT (LD_PRELOAD / DLL Forwarding):                                 |
|     - Provide drop-in replacements for libcuda.so.1 / nvcuda.dll and libcudart.so.           |
|     - Intercept symbol lookups (dlsym / GetProcAddress) for cuLaunchKernel, cuMemAlloc.       |
|                                                                                               |
|  2. PTX TO TARGET-NEUTRAL IR TRANSLATION:                                                     |
|     - When cuModuleLoadData(void* image) is called with PTX source:                           |
|       1. Parse PTX into CUDA_Translator Cross-Arch IR.                                        |
|       2. Execute normalization passes (WarpToWavefront, LDSNormalizer, MatrixAccelerator).    |
|       3. Lower IR to target bytecode: SPIR-V (Vulkan) or HIP C++ (AMD ROCm).                 |
|                                                                                               |
|  3. ZERO-COPY BUFFER MAPPING:                                                                 |
|     - cuMemAlloc(ptr, bytes) -> vkAllocateMemory (Vulkan) or hipMalloc (ROCm).               |
|     - Returns a 64-bit device address handle indistinguishable from a native CUDA pointer.    |
|                                                                                               |
|  4. STREAM EXECUTION:                                                                         |
|     - cuLaunchKernel(...) -> Enqueue VkCommandBuffer (Vulkan) or hipModuleLaunchKernel.       |
|     - Pushes kernel parameters into uniform push constants or device-mapped argument buffers. |
+-----------------------------------------------------------------------------------------------+
```

### Execution Flow: From `cuLaunchKernel` to Vulkan Compute Shader

```
Application calls:
cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, args, NULL)
  │
  ▼
[CUDA_Translator Interceptor: libcuda.so]
  │
  ├── 1. Lookup cached compiled pipeline for `func` handle
  │      └── If miss: Parse PTX AST -> Lower to SPIR-V -> Create VkPipeline
  │
  ├── 2. Map Stream to Vulkan Command Queue
  │      └── vkBeginCommandBuffer(stream->cmd_buffer)
  │
  ├── 3. Bind Push Constants & Descriptor Sets
  │      └── Pack `args` void** array into Vulkan push constant buffer (up to 128/256 bytes)
  │
  ├── 4. Record Dispatch Command
  │      └── vkCmdDispatch(stream->cmd_buffer, gridX, gridY, gridZ)
  │
  └── 5. Submit to Hardware Doorbell Register
         └── vkQueueSubmit(device->compute_queue, 1, &submit_info, fence)
```

---

## 5. Summary of Engineering Principles

1. **Keep the OS Kernel Pristine**: Never attempt to write a kernel driver for compute translation. Target user-space driver interfaces (`libcuda.so`/`nvcuda.dll`) to ensure safety, portability, and zero-privilege operation.
2. **Dynamic JIT Translation**: Treat PTX source as an input byte stream. Parse, optimize, and translate it to target-native SPIR-V or ROCm bytecodes at runtime.
3. **Preserve Memory Space Semantics**: Accurately map CUDA Global, Shared (LDS), Local (register spill), and Constant memory to target Vulkan or HIP storage classes to maintain memory coherence.
4. **Hardware Warp Coalescing**: Automatically handle the 32-lane warp vs. 64-lane wavefront divergence across AMD CDNA and Intel Xe architectures through the `WarpToWavefrontPass`.
