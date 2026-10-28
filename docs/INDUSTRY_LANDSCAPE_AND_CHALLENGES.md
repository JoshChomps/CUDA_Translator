# Industry Landscape, Existing Projects & Technical Gaps

This document provides an exhaustive survey of existing efforts to break the proprietary NVIDIA CUDA ecosystem ("the CUDA moat"), analyzing their architectural trade-offs, limitations, and the fundamental gaps that remain unsolved in the industry.

---

## 1. Competitive & Historical Landscape

Attempts to run CUDA software on non-NVIDIA hardware fall into two distinct philosophical approaches: **Runtime Interception/Translation** and **Compiler/IR Retargeting**.

| Project | Ingestion Level | Target Backend | Workflow & User Experience | Status & Major Roadblocks |
| :--- | :--- | :--- | :--- | :--- |
| **ZLUDA**<br>*(Andrzej Janik)* | **Runtime Interception**<br>(Drop-in `nvcuda.dll` / `libcuda.so`) | AMD ROCm / HIP<br>*(previously Intel oneAPI)* | **Zero recompilation required**.<br>Drop-in replacement for pre-built binaries. | Funded by Intel, then quietly funded by AMD (funding pulled in 2024). Open-source community fork (v6 in 2026). Struggles with binaries shipping only SASS and proprietary `cu*` libraries. |
| **SCALE**<br>*(Spectral Compute)* | **Source-Level Compiler**<br>(Drop-in `nvcc` replacement) | AMD ROCm / RDNA / CDNA | **Requires source code**.<br>Compiles native `.cu` directly to AMDGPU machine code. | Commercial closed-source enterprise product (free for academic/evaluation). Cannot handle pre-compiled third-party binaries. |
| **CuPBoP**<br>*(Purdue & Georgia Tech)* | **LLVM IR Transformation**<br>(NVVM IR) | AMD GPUs & Heterogeneous CPUs | **Requires compiler hook**.<br>Translates NVVM IR into HIP/OpenCL IR. | Academic research framework. Highly modular, but requires building from source via Clang/NVVM tooling. |
| **HIPIFY**<br>*(AMD)* | **Source AST / Regex**<br>(`hipify-clang`, `hipify-perl`) | AMD HIP C++ | **Manual source porting tool**.<br>Performs syntactic text replacement. | Official AMD tool. Breaks on inline PTX assembly, complex warp ballot logic, and memory-bank assumptions. |
| **DPC++ / DPCT**<br>*(Intel oneAPI)* | **Source AST Translation** | SYCL / oneAPI | **One-time code migration**.<br>Converts CUDA C++ to SYCL. | Requires extensive manual developer cleanup after translation. |
| **Coriander** | **Source / IR** | OpenCL 1.2 | Recompilation targeting standard OpenCL. | Limited to older OpenCL 1.2 features; lacks modern subgroup and tensor core acceleration. |

---

## 2. Fundamental Technical Challenges & Industry Gaps

Why is a universal, seamless "Wine for CUDA" so difficult to achieve? The industry faces four major architectural hurdles:

### A. The SASS vs. PTX Binary Wall
* **PTX (Parallel Thread Execution)**: A virtual, forward-compatible assembly representation. Translating PTX is relatively clean because register allocations, memory spaces, and control flow are well-defined.
* **SASS (Source-Associated Assembly)**: Real machine instructions emitted by `nvcc` for specific GPU microarchitectures (e.g., `sm_80`, `sm_90`) and packed into ELF `.cubin` containers.
* **The Gap**: Modern commercial software and official PyTorch binary wheels often strip PTX to minimize binary footprint and protect proprietary algorithms. Decompiling undocumented, non-ABI-stable SASS microcode back into an Intermediate Representation is a massive reverse-engineering undertaking.

### B. Microarchitecture SIMT Divergence

#### 1. Warp Size (32 Lanes) vs. Wavefront Size (64 Lanes)
* NVIDIA hardware uses a fixed **32-thread Warp**.
* AMD Datacenter GPUs (CDNA, MI200, MI300) use a **64-thread Wavefront**.
* **The Dilemma**:
  * If a translator masks out the upper 32 threads (`EXEC[32..63] = 0`), it achieves semantic safety but wastes **50% of the compute hardware**.
  * If a translator coalesces two warps ($W_0$ and $W_1$) into one wavefront, any divergent branch taken by $W_0$ but not $W_1$ serializes execution across both, causing catastrophic throughput loss.

#### 2. Independent Thread Scheduling (ITS)
* Introduced in NVIDIA Volta (SM 7.0), ITS gives each individual thread within a warp its own Program Counter (PC) and call stack.
* Prior to ITS, warps executed strictly in lockstep with a shared SIMT divergence stack.
* Many modern CUDA algorithms rely on implicit warp-synchronous programming or fine-grained divergence loops. When translated to architectures that lack hardware ITS, these algorithms risk **live-locks, deadlocks, or subtle data races**.

#### 3. Memory Consistency Models
* PTX defines a sophisticated memory consistency model with explicit scopes: `.sys` (system), `.gpu` (device), and `.cta` (workgroup/block), alongside release/acquire semantics.
* Mapping these to target memory models (e.g., Vulkan memory model or AMD flat address spaces) without inserting overly conservative global cache flushes is critical to avoiding bandwidth collapse.

### C. The Proprietary Closed Ecosystem (`cu*` Libraries)
* Real-world AI applications (vLLM, TensorRT-LLM, PyTorch) rarely execute raw CUDA kernels in isolation. They rely heavily on NVIDIA's proprietary closed-source libraries:
  * **cuBLAS**: Highly tuned GEMM routines.
  * **cuDNN**: Deep neural network convolution and normalization primitives.
  * **NCCL**: High-performance multi-GPU collective communications.
  * **CUTLASS**: Template-based abstractions for Tensor Core GEMM.
* Translating a kernel launch succeeds only to crash on the first call to `cudnnCreate()` or `cublasSgemm()`. A viable translation layer must provide ABI-compatible forwarding shims to target libraries (e.g., `rocBLAS`, `MIOpen`, `RCCL`).

### D. The Performance Parity Gap ("The Leaky Abstraction")
* High-performance GPU kernels are microarchitecturally co-designed with specific hardware boundaries:
  * Register pressure thresholds (e.g., 32 vs. 64 registers per thread before spilling to global memory).
  * Cache line widths (NVIDIA 128-byte cache lines vs. AMD 64-byte lines).
  * Shared memory bank layout (32 banks with 4-byte boundaries).
* A naive, line-by-line translated kernel frequently incurs 30% to 70% performance degradation due to unexpected register spills, LDS bank serialization, and misaligned global memory transactions.
