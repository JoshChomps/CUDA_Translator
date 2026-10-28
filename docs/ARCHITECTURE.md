# Technical Architecture & Internal Compiler Design

This document details the internal design of the **CUDA_Translator** framework, describing the Intermediate Representation (IR), transformation pipelines, and code-generation backends.

---

## 1. Neutral GPU Intermediate Representation (Cross-Arch IR)

The core principle of `CUDA_Translator` is decoupling input representations from vendor-specific hardware microarchitectures. The compiler achieves this using an SSA-based Intermediate Representation modeling GPU compute paradigms.

### Memory Spaces
The IR explicitly models the distinct physical memory hierarchies found on GPUs:
* `Global`: High-bandwidth device DRAM accessible by all workgroups.
* `Shared` (LDS): High-speed, on-chip scratchpad memory shared among threads of a single workgroup.
* `Local`: Thread-private stack storage (register spills).
* `Constant`: Read-only uniform memory cached at the compute unit level.
* `Param`: Kernel launch argument buffer.

### Type System
The IR supports strong typing:
* Primitive floating-point: `f16`, `f32`, `f64`, `bf16`.
* Integers: `u8`, `u16`, `u32`, `u64`, `s8`, `s16`, `s32`, `s64`.
* Predicates: `pred` (1-bit condition flags used in GPU branch predication).
* Pointers: `ptr<T, Space>`.
* Matrix Tiles: `cooperative_matrix<T, M, N, K, Layout>` for tensor acceleration.

---

## 2. Transformation Pipeline & Normalization Passes

```
[PTX AST] 
    │
    ▼
[Raw GPU IR]
    │
    ├─► Pass 1: WarpToWavefrontPass
    │     ├── Remap thread indices: tid.x -> local_id.x
    │     ├── Clamp intra-warp shuffle boundaries to 32 lanes
    │     └── Transform 32-bit activemask ballots to 64-bit/uvec4
    │
    ├─► Pass 2: LDSNormalizerPass
    │     ├── Scan shared memory access patterns
    │     ├── Detect potential bank collisions (e.g. 32-word stride)
    │     └── Inject array padding: [M][N] -> [M][N + padding]
    │
    ├─► Pass 3: MatrixAcceleratorPass
    │     ├── Detect WMMA intrinsics (wmma.load, wmma.mma, wmma.store)
    │     └── Lower to Cooperative Matrix operations (MFMA / VK_KHR_cooperative_matrix)
    │
    ▼
[Optimized Target-Neutral IR]
    │
    ├─► Vulkan Backend  ──► Vulkan Compute Shader (GLSL/SPIR-V)
    └─► HIP Backend     ──► AMD ROCm / HIP C++ Kernel
```

### A. Warp-to-Wavefront Transformation
NVIDIA execution scheduling groups 32 threads into a warp. AMD architectures group threads into wavefronts of either 32 (RDNA) or 64 (CDNA/GCN).

1. **Active Thread Masking**:
   On 64-thread wavefront targets, when running uncoalesced kernels, the pass injects a lane guard:
   ```c
   // Only lower 32 threads perform work
   if (lane_id < 32) {
       // execute warp operations
   }
   ```
2. **Shuffle Remapping**:
   In CUDA, `__shfl_sync(mask, val, lane)` communicates exclusively within 32 lanes. On 64-thread wavefronts, a naive cross-lane permute might access data from lane $k+32$. The pass rewrites shuffles to partition lanes:
   ```c
   uint sub_warp_lane = lane_id % 32;
   uint target_lane = (sub_warp_lane + delta) % 32 + (lane_id - sub_warp_lane);
   ```

### B. Shared Memory & LDS Normalizer
GPU shared memory is partitioned into memory banks (typically 32 banks on both NVIDIA SMs and AMD CUs, but with differing bank-conflict resolution and stride policies). When code writes data with stride 32, every thread accesses bank 0, causing a 32-way conflict and serialized execution.

The `LDSNormalizerPass`:
1. Inspects multi-dimensional shared memory declarations (`.shared .f32 smem[32][32]`).
2. Automatically pads the inner dimension by 1 element (`.shared .f32 smem[32][33]`).
3. Updates all address calculations indexing into the shared memory array.

### C. Matrix Acceleration Routing
CUDA expresses Tensor Core matrix multiplies through `nvcuda::wmma` or PTX `wmma.mma.sync`.
The `MatrixAcceleratorPass`:
* Ingests fragments: `wmma.load.a`, `wmma.load.b`, `wmma.load.c`.
* Lowers to `coopmat` types:
  - In Vulkan: `coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>`.
  - In HIP: `rocwmma::fragment` or inline AMD `__builtin_amdgcn_mfma`.

---

## 3. Runtime Interception Layer Architecture

To run existing CUDA binaries without source recompilation, the runtime interceptor replaces `nvcuda.dll` (Windows) or `libcuda.so` (Linux).

1. **Context & Device Virtualization**:
   - `cuInit(0)` initializes target Vulkan or ROCm hardware devices.
   - `cuDeviceGet()` queries target hardware capabilities and reports them using normalized CUDA compute capability values (e.g., reporting SM 8.0 for modern AMD RDNA/CDNA).
2. **Memory Virtualization**:
   - `cuMemAlloc(d_ptr, bytes)` allocates host-accessible device memory (e.g., Vulkan `VkBuffer` with `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` or `hipMalloc`).
   - Virtual pointers are tracked in an internal page table mapping 64-bit virtual addresses to native device buffer descriptors and offsets.
3. **Module & Kernel Dispatch**:
   - `cuModuleLoadData(module, ptx_code)` invokes the translation pipeline in-memory, compiling the PTX to Vulkan SPIR-V or ROCm code in real time.
   - `cuLaunchKernel(...)` binds descriptor sets, uploads parameters to a uniform buffer, and records a compute dispatch command buffer.
