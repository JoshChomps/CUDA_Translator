# CUDA_Translator: Cross-Architecture GPU Translation Layer

[![Status: Alpha Prototype](https://img.shields.io/badge/status-alpha--prototype-blue.svg)](SPECIFICATION.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Target: AMD ROCm / Vulkan](https://img.shields.io/badge/targets-AMD%20ROCm%20%7C%20Vulkan%20SPIR--V-orange.svg)](SPECIFICATION.md)

A hardware-agnostic compiler and runtime translation framework designed to ingest NVIDIA CUDA workloads (PTX assembly and Driver API calls), translate them into a neutral Intermediate Representation (IR), apply microarchitectural hardware normalization passes, and lower them cleanly to **AMD ROCm/HIP** and **Universal Vulkan Compute (SPIR-V)**.

---

## Key Features

* **Compiler-Level Translation**: Operates directly on PTX virtual ISA and GPU IR rather than fragile source-code regex replacement.
* **Warp (32) $\to$ Wavefront (64/32) Abstractor**: Normalizes thread scheduling, handles active thread masking, and converts warp shuffle intrinsics (`__shfl_sync`, ballot masks) into target subgroup/wavefront primitives.
* **Virtual LDS / Shared Memory Cache Normalizer**: Injects stride padding and memory boundary alignment to eliminate bank conflicts across differing hardware cache configurations.
* **Tensor Core $\to$ Matrix Core Routing**: Intercepts `wmma` matrix operations and lowers them to AMD MFMA / Vulkan `VK_KHR_cooperative_matrix`.
* **Universal Target Support**: Emits Vulkan Compute Shaders executable across AMD, Intel, and Apple GPUs, as well as native AMD HIP/ROCm C++ code.

---

## System Architecture

```
                       +-----------------------------------+
                       |      CUDA Kernel / Host Code      |
                       |  (PTX Assembly / Driver API Calls)|
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       |         PTX Frontend & AST        |
                       |   Tokenization, Type Ingestion    |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       |     Neutral GPU Intermediate      |
                       |        Representation (IR)        |
                       | (SSA CFG, Memory Spaces, Subgroups)|
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
          +-------------------+ +------------------+ +------------------+
          | Warp-to-Wavefront | |  LDS / Cache Bank| | Tensor-to-Matrix |
          | Normalization     | |  Normalizer Pass | | Routing Pass     |
          +-------------------+ +------------------+ +------------------+
                    |                    |                    |
                    +--------------------+--------------------+
                                         |
                                         v
                       +-----------------------------------+
                       |         Backend Emitters          |
                       | +-------------------------------+ |
                       | | Vulkan Compute Shader (GLSL)  | |
                       | | AMD HIP / ROCm Kernel (C++)   | |
                       | +-------------------------------+ |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Hardware Execution on Target GPU  |
                       |  (AMD Radeon/Instinct, Intel, etc)|
                       +-----------------------------------+
```

---

## 📂 Repository Layout

```
CUDA_Translator/
├── SPECIFICATION.md             # In-depth architectural specification and mathematical formulations
├── README.md                    # Project overview and quickstart
├── pyproject.toml               # Project configuration and dependency specification
├── docs/
│   ├── ARCHITECTURE.md          # Technical deep-dive on IR and pass pipelines
│   ├── INDUSTRY_LANDSCAPE_AND_CHALLENGES.md # Survey of ZLUDA, SCALE, CuPBoP, and technical gaps
│   └── GOOGLE_ECOSYSTEM_AND_TPU_BACKEND.md  # Google TPUs (v2-v6e), Pallas, Mosaic, and backend design
├── cuda_translator/             # Core translation engine
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface for kernel translation
│   ├── ir/                      # Neutral GPU Intermediate Representation
│   │   ├── __init__.py
│   │   ├── types.py             # Scalar, vector, pointer, and matrix types
│   │   ├── instructions.py      # SSA instruction definitions
│   │   └── module.py            # Kernel, BasicBlock, and Module structures
│   ├── frontend/                # Source ingestion
│   │   ├── __init__.py
│   │   ├── ptx_lexer.py         # PTX tokenizer
│   │   └── ptx_parser.py        # PTX to Neutral IR parser
│   ├── transform/               # Microarchitectural normalization passes
│   │   ├── __init__.py
│   │   ├── warp_to_wavefront.py # 32-to-64 thread mapping & shuffle remapping
│   │   ├── lds_normalizer.py    # Shared memory bank conflict avoidance
│   │   └── matrix_accelerator.py# WMMA to Cooperative Matrix conversion
│   ├── backend/                 # Target code generation
│   │   ├── __init__.py
│   │   ├── vulkan_backend.py    # Vulkan Compute Shader / SPIR-V generator
│   │   └── hip_backend.py       # AMD ROCm/HIP C++ generator
│   └── runtime/                 # Interception and driver emulator
│       ├── __init__.py
│       ├── driver_api.py        # Python CUDA Driver API emulator
│       ├── include/
│       │   └── cuda.h           # Standard CUDA Driver API header stubs
│       └── interceptor.c        # C runtime hook for dynamic library interception
├── benchmarks/
│   └── kernels/                 # Reference test kernels
│       ├── vector_add.ptx       # Basic arithmetic & memory access
│       ├── warp_shuffle.ptx     # Intra-warp reduction (__shfl_sync)
│       └── matrix_wmma.ptx      # Matrix multiplication (WMMA Tensor Core)
└── tests/                       # Automated test suite
    ├── test_ptx_parser.py       # Parser verification
    ├── test_transforms.py       # Transformation passes verification
    ├── test_backends.py         # Vulkan & HIP emission verification
    └── test_cli.py              # CLI integration tests
```

---

## Quickstart

### 1. Requirements
* Python 3.10+
* (Optional for execution) Vulkan SDK or AMD ROCm

### 2. Installation
```powershell
git clone https://github.com/your-username/CUDA_Translator.git
cd CUDA_Translator
pip install -e .
```

### 3. Translating a PTX Kernel
To translate a CUDA PTX kernel into a universal Vulkan Compute Shader:
```powershell
python -m cuda_translator.cli translate benchmarks/kernels/vector_add.ptx --target vulkan -o vector_add.comp
```

To translate into an AMD HIP C++ kernel with Wavefront-64 remapping:
```powershell
python -m cuda_translator.cli translate benchmarks/kernels/warp_shuffle.ptx --target hip -o warp_shuffle.hip.cpp
```

### 4. Running the Test Suite
```powershell
pytest -v tests/
```

---

## 🛣️ Roadmap

- [x] **Milestone 1**: PTX Lexer/Parser, Neutral GPU IR, and core data structures.
- [x] **Milestone 2**: Hardware transformation passes (Warp-to-Wavefront, LDS cache bank normalizer, Tensor-to-Matrix router).
- [x] **Milestone 3**: Universal Vulkan GLSL/SPIR-V and AMD HIP code generators.
- [ ] **Milestone 4**: Direct LLVM Bitcode (`.bc`) emission and integration with `clang` / `AMDGPU` backend.
- [ ] **Milestone 5**: Full transparent C/C++ dynamic library loader (`nvcuda.dll` / `libcuda.so` replacement).
- [ ] **Milestone 6**: Google TPU Backend (Pallas / Mosaic via XLA) for running CUDA/PTX on TPU pods.
