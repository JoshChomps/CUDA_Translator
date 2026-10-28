# Google Compute Ecosystem & TPU Translation Layer Blueprint

This document analyzes Google's proprietary compute hardware, examines how Google structures its accelerator software stack, and details an architectural blueprint for extending **CUDA_Translator** with a **Google TPU Backend (Pallas / Mosaic)**.

---

## 1. Does Google Have Their Own Version of CUDA?

**Google does not manufacture discrete desktop or datacenter GPUs.** Instead, Google designs and deploys custom ASICs called **TPUs (Tensor Processing Units)**:

* **TPU v2 / v3**: First widely deployed multi-core training and inference pods.
* **TPU v4**: Optical Circuit Switch (OCS) interconnect for multi-thousand-chip clusters.
* **TPU v5e / v5p**: Cost-optimized and high-performance chips for LLM training.
* **TPU v6e ("Trillium")**: Latest generation featuring 4.7x compute density over v5e.

---

## 2. GPU vs. TPU: Architectural Differences

| Dimension | NVIDIA GPU (Ampere, Hopper, Blackwell) | Google TPU (v4, v5p, v6e Trillium) |
| :--- | :--- | :--- |
| **Execution Paradigm** | **SIMT (Single Instruction, Multiple Threads)**<br>Thousands of threads grouped into 32-lane warps. | **Systolic Array & VLIW (Very Long Instruction Word)**<br>Hardware matrix units (MXUs) stream data through 2D grids of ALUs. |
| **Memory Hierarchy** | Registers $\to$ Shared Memory (LDS) $\to$ L1/L2 Caches $\to$ HBM. | Scalar Memory (**SMEM**) $\to$ Vector Memory (**VMEM**) $\to$ High-Bandwidth Memory (**HBM**). |
| **Memory Management** | Automatic hardware caching + developer-managed Shared Memory. | **Explicit software-scheduled DMA transfers** between HBM and VMEM. |
| **Matrix Operations** | Tensor Cores (e.g. 16x16x16 tile WMMA). | Matrix Multiply Unit (**MXU**): Fixed 128x128 systolic matrix engines. |

---

## 3. Google's Accelerator Software Stack

Historically, TPUs were "black boxes" accessible only through high-level graph compilers. Developers wrote TensorFlow or JAX code, and **XLA (Accelerated Linear Algebra)** lowered the whole graph into hardware micro-ops. Developers could *not* write custom hardware kernels.

To compete with custom GPU kernels (e.g., FlashAttention, custom MoE routers, and FP8 kernels), Google introduced a low-level kernel programming stack:

### A. Pallas (`jax.experimental.pallas`)
* **Google's equivalent to CUDA / Triton for TPUs.**
* An embedded Python domain-specific language (DSL) within JAX.
* Exposes explicit control over:
  1. **Grid Tiling**: Dividing tensors into blocks assigned to individual TPU cores.
  2. **Memory Allocation**: Placing intermediate arrays in high-speed on-chip **VMEM**.
  3. **DMA Pipelining**: Overlapping data transfer from HBM into VMEM with computation.

### B. Mosaic (`jax.experimental.mosaic`)
* The underlying low-level compiler framework and MLIR dialect (`mosaic`).
* Translates Pallas and JAX kernel graphs directly into TPU machine micro-ops.

### C. OpenXLA & Tokamax
* **OpenXLA**: The open-source compiler standard for heterogeneous AI accelerators (GPUs, TPUs, NPUs).
* **Tokamax**: Google's open accelerator library providing optimized kernels (FlashAttention, MoE) implemented across both GPUs (via Triton) and TPUs (via Pallas/Mosaic).

---

## 4. Architectural Blueprint: PTX $\to$ Google TPU (Pallas / Mosaic) Backend

Extending `CUDA_Translator` to target Google TPUs represents an immense strategic opportunity. It allows the vast ecosystem of PyTorch CUDA kernels to run on Google Cloud TPUs without requiring manual developer rewrites.

```
                           +-------------------------------+
                           |      CUDA Kernel / PTX        |
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           |    CUDA_Translator Neutral    |
                           |    GPU Intermediate Rep (IR)  |
                           +---------------+---------------+
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
                    v                      v                      v
          +--------------------+ +--------------------+ +--------------------+
          | CTA to Grid Tiling | | Shared Mem to VMEM | | WMMA to MXU Tile   |
          | Transform Pass     | | DMA Normalizer Pass| | Lowering Pass      |
          +--------------------+ +--------------------+ +--------------------+
                    |                      |                      |
                    +----------------------+----------------------+
                                           |
                                           v
                           +-------------------------------+
                           |     Google TPU Backend        |
                           |  (JAX / Pallas Python Module  |
                           |   or Mosaic MLIR Dialect)     |
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           | Execution on Google Cloud TPU |
                           |  (TPU v4, v5e, v5p, Trillium) |
                           +-------------------------------+
```

### Key Transformation Passes Required for TPU Target

1. **CTA Grid $\to$ Pallas Tiling Pass**:
   * CUDA assigns threads into blocks: `dim3 gridDim`, `dim3 blockDim`.
   * The pass maps the kernel grid into Pallas `GridSpec`:
     ```python
     grid = (grid_dim_x, grid_dim_y)
     in_specs = pl.BlockSpec(lambda i, j: (i * block_size_m, j * block_size_k), (block_size_m, block_size_k))
     ```
2. **Shared Memory $\to$ VMEM & Asynchronous DMA Pass**:
   * CUDA `.shared` arrays map to Pallas `VMEM` scratchpad buffers.
   * Explicit global load/store instructions (`ld.global`, `st.global`) are lifted into asynchronous block DMA transfers.
3. **WMMA $\to$ MXU Systolic Lowering Pass**:
   * Identifies 16x16x16 CUDA Tensor Core operations and lowers them to TPU 128x128 systolic matrix multiplications or `jnp.dot` primitives executed by the MXU.
