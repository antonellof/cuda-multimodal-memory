# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project overview

**MARS (Memory for Autonomous Real-time Systems)** — a GPU-resident multimodal memory
substrate for **real-time AI systems operating in the physical world**:
autonomous vehicles, humanoid robots, AR/VR headsets, voice agents, drone
swarms, surgical robots, live perception pipelines. Text, audio, image,
proprioceptive, and other sensor embeddings share a single 768-D space and
live as nodes in a Neural Shortcut Network (NSN) with cross-modal bridges.
Retrieval runs as four CUDA kernels entirely on GPU-resident data.

### Mission

> Investigate whether a GPU-resident memory substrate can meet the hard
> real-time retrieval deadlines of embodied AI systems while supporting
> native cross-modal queries and temporal decay within a single unified
> index.

### Goals

- **Goal 1**: Meet the deadline budgets of real-time workloads. p99 under
  1 ms at 60 Hz for AV perception, under 1 ms at 1 kHz for humanoid
  control, under 5 ms at 90 Hz for AR/VR, under 20 ms at 30 Hz for voice.
- **Goal 2**: Be the first general-purpose alternative to the bespoke C++
  circular buffers currently shipping in production autonomy and robotics
  stacks at Waymo, Figure, Apple Vision Pro, etc.
- **Goal 3**: Support native multimodality as a first-class feature via
  cross-modal graph bridges, not as a bolt-on per-modality index.
- **Goal 4**: Keep the code genuinely runnable and deadline-checked:
  `make tests` passes host-only today; `make && make check` on any
  vast.ai A100 instance produces real hardware numbers in under a minute;
  `make bench-av` / `-robot` / `-ar` / `-voice` exit non-zero if the
  budget is missed.

### Non-goals

- **Not** a replacement for Postgres/pgvector or any other vector database.
  Same conceptual layer (indexing, similarity search, retrieval) but
  different latency envelope, different durability model, different
  deployment target. Think cuBLAS vs LAPACK — same operations, different
  hardware. They are complementary, not competing.
- **Not** an archival store. Working set is seconds to hours of recent
  sensor data, bounded to fit in GPU VRAM. For billion-record archives,
  use a vector DB.

## Current status (honest assessment)

### ✅ Done and verified
- **Four CUDA kernels** (cosine similarity, temporal rerank, top-K,
  warp-cooperative BFS). All syntax-verified; host C++ compiles cleanly
  under g++.
- **Multimodal NSN graph** with cross-modal bridges (ring lattice + skip
  connections + hub supernodes + Watts-Strogatz rewiring + modality bridges).
- **Host-only unit test suite** (`tests/test_memory_graph.cpp`) — 7 tests
  verifying CSR validity, edge symmetry, no self-loops, cross-modal bridge
  invariant, L2 normalization, degree bounds, synthetic counts.
  **All 7 passing.** Run with `make tests` — no GPU needed.
- **Four application demonstrators** under `demos/` for the real-world
  target workloads:
  - `av_perception/` — 60 Hz, p99 < 1 ms, ~2,400 memories (AV perception)
  - `robot_episodic/` — 1 kHz, p99 < 1 ms, ~6,000 memories (humanoid control)
  - `ar_spatial/` — 90 Hz, p99 < 5 ms, ~20,000 memories (AR/VR SLAM)
  - `voice_agent/` — 30 Hz, p99 < 20 ms, ~3,000 memories (voice agent)
  Each is runnable CUDA C++ with its own success criterion.
- **Deadline-aware latency benchmark** (`src/latency_bench.cu`) — runs at
  a fixed sensor rate, reports p50/p90/p99/p99.9/p99.99/max/jitter/miss
  rate, emits JSON, exits non-zero if p99 exceeds budget. Presets for
  all four demos via `make bench-av` / `-robot` / `-ar` / `-voice`.
- **Validation harness** (`src/validate.cu`) — JSON output with per-stage
  timings, GPU device properties, recall@10 correctness check.
- **CPU reference implementation** (`scripts/bench_cpu.py`) for projection
  baselines.
- **Architecture diagrams** (4 PNGs in `docs/`).
- **arXiv paper** (`paper/main.tex`) — 17 pages, compiles cleanly via
  `pdflatex && bibtex && pdflatex && pdflatex`. Uses `\placeholder{...}`
  macro for all unmeasured numbers (rendered red/bold). Structure:
  mission → target workloads → architecture → NSN → kernels →
  demonstrators → methodology → results → comparison → limitations.
- **Makefile** with targets for engine, validate, latency, demos, tests,
  and all the `bench-*` and `demo-*` convenience targets. Multi-arch
  build (sm_70..sm_89).

### ⚠️ Partially done
- **GPU numbers measured through v2 pipeline.** A100X results in
  `results/` folder. v3 pipeline (device-driven BFS, GPU-side result
  compaction, pinned host memory, GPU keepalive) is implemented and
  needs re-validation on hardware.
- **`src/engine_server.cu` exists but is untested on GPU.**

### ❌ Not started
- Python bindings (pybind11).
- FP16 + tensor-core optimization of the similarity kernel.
- CUDA Graph capture of the 4-kernel pipeline.
- Multi-GPU sharding via NVLink.
- Persistence layer (checkpoint/restore of the memory graph to disk).
- Streaming insertion API.

## Repository layout

```
cuda-multimodal-memory/
├── CLAUDE.md                     ← you are here
├── README.md                     ← public-facing project page
├── LICENSE                       ← MIT
├── CONTRIBUTING.md
├── Makefile                      ← nvcc build, `make check` runs validate
├── .gitignore
│
├── include/
│   ├── memory_graph.h            ← multimodal CSR graph, host side
│   └── memory_cuda.cuh           ← kernel declarations, DeviceMemoryGraph, RetrievalConfig
│
├── src/
│   ├── memory_graph.cpp          ← NSN construction (5 phases + multimodal bridges)
│   ├── memory_cuda.cu            ← the 4 CUDA kernels + query_memory() driver
│   ├── main.cu                   ← interactive benchmark CLI
│   ├── validate.cu               ← JSON-emitting validation harness for vast.ai
│   ├── latency_bench.cu          ← deadline-aware latency benchmark
│   └── engine_server.cu          ← JSON-line REPL for Python adapters
│
├── scripts/
│   ├── bench_cpu.py              ← NumPy reference pipeline (for projections)
│   └── make_diagrams.py          ← regenerates the 4 architecture diagrams
│
├── docs/
│   ├── ARCHITECTURE.md           ← deep dive on the 4 kernels
│   ├── HARDWARE_VALIDATION.md    ← ⚠ TRUNCATED — needs completion
│   ├── thesis.docx               ← original thesis (to be replaced by paper/main.tex)
│   ├── diag_pipeline.png         ← Figure 1
│   ├── diag_graph.png            ← Figure 2
│   ├── diag_kernels.png          ← Figure 3
│   └── diag_benchmark.png        ← Figure 4
│
├── paper/
│   ├── main.tex                  ← arXiv-format LaTeX (17 pages, compiles to PDF)
│   ├── main.pdf                  ← compiled paper
│   ├── references.bib
│   └── figures/                  ← copies of docs/*.png
│
│
└── benchmark.json                ← raw CPU-reference numbers from scripts/bench_cpu.py
```

## Build and test commands

### Building
```bash
make                # builds both memory_engine and validate
make engine         # just the benchmark CLI
make validate       # just the validation harness
make clean          # remove build artifacts
make info           # print nvcc + GPU info (useful on vast.ai to verify setup)
```

### Running locally (requires NVIDIA GPU)
```bash
./memory_engine 4000 2000 2000 768    # interactive benchmark
./validate 100 768                     # hardware validation, prints JSON to stdout
make check                             # runs validate and saves to results.json
```

### Syntax checks (no GPU needed)
The codebase uses a CUDA runtime stub in `/tmp/cuda_stub/` (not in the repo)
so that `.cu` files can be syntax-checked with plain g++. This is useful when
iterating without a GPU. The stub is a set of empty/no-op declarations for
`cudaMalloc`, `__shfl_down_sync`, `atomicCAS`, etc.

To set it up in a fresh environment, create `/tmp/cuda_stub/cuda_runtime.h`
with stubs for: `cudaError_t`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`,
`cudaMemset`, `cudaEvent*`, `cudaDeviceProp`, `cudaGetDeviceProperties`,
`cudaMemGetInfo`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`, plus the
`__global__`/`__device__`/`__shared__`/`__restrict__` macros as no-ops.

Then:
```bash
# Host-only C++ files
g++ -std=c++17 -Iinclude -fsyntax-only -c src/memory_graph.cpp

# CUDA .cu files (will show <<<>>> launch errors — expected)
g++ -std=c++17 -Iinclude -I/tmp/cuda_stub -fsyntax-only -x c++ src/memory_cuda.cu
g++ -std=c++17 -Iinclude -I/tmp/cuda_stub -fsyntax-only -x c++ src/validate.cu
g++ -std=c++17 -Iinclude -I/tmp/cuda_stub -fsyntax-only -x c++ src/main.cu
```

All files should pass cleanly except for `<<<blocks, threads>>>` kernel-launch
syntax in `memory_cuda.cu`, which only nvcc understands.

## Coding conventions

**C++ / CUDA:**
- C++17 host code, CUDA C++ for device code
- 4-space indent, no tabs
- `snake_case` for functions and variables, `CamelCase` for types
- Explicit integer types (`int32_t`, `int64_t`) — avoid bare `int`
- `__restrict__` on all pointer arguments to `__global__` kernels
- Every kernel has a comment block above it explaining grid/block layout
- Wrap CUDA runtime calls in `CUDA_CHECK()` from `include/memory_cuda.cuh`
- Ship both a thread-centric and a warp-centric variant when the latter helps

**Python:**
- Python 3.9+
- Type hints throughout
- `dataclass` for structured records
- Context managers (`__enter__`/`__exit__`) for resource-owning classes
- No heavy frameworks — stdlib + numpy + sentence-transformers only

## Architecture — the parts Claude needs to know

### Data layout
The memory graph is stored in **Compressed Sparse Row (CSR)** format on device:

| Array         | Size    | Type     | Purpose                          |
|---------------|---------|----------|----------------------------------|
| `row_offsets` | N+1     | int32    | Prefix-sum of node degrees       |
| `col_indices` | E       | int32    | Neighbor node IDs (sorted)       |
| `embeddings`  | N × D   | float32  | L2-normalized embedding vectors  |
| `modalities`  | N       | int32    | 0=text, 1=audio, 2=image         |
| `timestamps`  | N       | float32  | For temporal decay               |

For N = 1M memories at D = 768 with avg degree 12: ~3.1 GB on device.

### The kernels (see ARCHITECTURE.md for full detail)

1. **`cosine_similarity_kernel`** — grid N blocks × 256 threads. Each block
   computes one dot product. Warp-shuffle reduction via `__shfl_down_sync`.
2. **`temporal_rerank_kernel`** — one thread per memory. `score * __expf(-λ·age)`.
3. **`top_k_kernel`** — single block, per-thread register heaps, shared-memory merge.
   Also: **`top_k_tiled_pass1/pass2`** — tiled two-pass variant used by
   `query_memory_fast()`. Each tile (4096 elements) produces a local top-K
   in parallel; a second pass merges tile results. Eliminates the serial
   thread-0 bottleneck that dominated at ~0.5 ms in the original.
4. **`bfs_expand_kernel`** — one warp per frontier node. `atomicCAS` for
   race-free neighbor claiming. This is the kernel that makes cross-modal
   retrieval work via the NSN.
5. **`bfs_init_seeds_kernel`** — GPU-side BFS seed initialization from
   the top-K output. Eliminates the D2H round trip between top-K and BFS.

### Two retrieval APIs

- **`query_memory()`** — original pipeline. Allocates/frees scratch buffers
  per query. Correct but has ~0.3 ms of overhead from `cudaMalloc`/`cudaFree`
  and synchronous D2H copies. Retained for backward compatibility.
- **`query_memory_fast()`** — optimized pipeline. Uses a pre-allocated
  `QueryContext` (created once via `create_query_context()`), tiled top-K,
  and GPU-side BFS initialization. All demos and benchmarks now use this path.

### The NSN construction (memory_graph.cpp)
Five phases, applied in order:
1. Ring lattice (k=6 local neighbors)
2. Hierarchical skip connections (powers of 2)
3. Hub supernodes at √N intervals with extra random links
4. Small-world rewiring (p=0.15)
5. **Cross-modal bridges** — every node gets one edge to each other modality

Phase 5 is the key innovation. Remove it and the system degenerates to a
unimodal graph + separate cross-modal index.

## Known pitfalls

### When editing CUDA kernels
- **`__shared__` arrays must have compile-time sizes.** Don't parameterize them.
- **`atomicCAS` on `int32_t*`** — CUDA's signature uses `int*`. Cast explicitly
  or make the arrays `int32_t` which usually matches `int`.
- **Warp shuffle masks** should be `0xFFFFFFFF` for full-warp ops; partial-warp
  shuffles need careful mask management.
- **`cudaMemcpyAsync` + pinned memory** — v3 uses `cudaMallocHost` for the
  query vector staging buffer in `QueryContext::h_query_pinned`.

### When editing the Makefile
- Multi-arch `-gencode` flags bloat the binary. sm_70..sm_89 covers V100 to
  RTX 40xx but not H100 (sm_90). Add `sm_90` only if targeting H100 explicitly.
- `-x cu -c` is used for `.cpp` files so they go through nvcc. This is
  necessary because they include `memory_graph.h` which is also included from
  `.cu` files and has to be compiled identically.

### When running on vast.ai
- Pick an image with CUDA 12+ preinstalled (`pytorch/pytorch:*-devel` works).
- `nvidia-smi` must show the GPU before `make` — if it doesn't, the container
  wasn't started with `--gpus all` or equivalent.
- The `make check` target takes under a minute; no need for a long-rental
  instance. Spot instances are fine.
- Destroy the instance immediately after — vast.ai bills by the second.

## Priority queue for next sessions

**Session N+1 (DONE — engine_server.cu written):**
1. ✅ Wrote `src/engine_server.cu` — JSON-line REPL with ADD, QUERY, DELETE,
   STATS, SAVE, LOAD, SHUTDOWN commands.
2. ✅ Updated Makefile with `make server` target.

**Session N+2 (DONE — hardware validation on vast.ai A100X):**
1. ✅ `make check` → `results.json` with sub-millisecond average latency.
2. ✅ `make demo-*` → all four demos pass their p99 deadlines at GPU kernel timing.
3. ✅ `make bench-*` → wall-clock latency benchmarks. AR/VR and voice pass;
   AV and robot exceed 1 ms budget by 12-41%.
4. ✅ Paper and README updated with all measured numbers.

**Session N+3 (DONE — optimized query path):**
1. ✅ Identified six specific bottlenecks in `query_memory()`.
2. ✅ Implemented `query_memory_fast()` with three fixes:
   - Pre-allocated `QueryContext` (no per-query cudaMalloc).
   - Tiled two-pass top-K kernel (eliminates 0.5 ms serial bottleneck).
   - GPU-side BFS initialization kernel (eliminates D2H round trip).
3. ✅ All demos and benchmarks switched to `query_memory_fast()`.
4. ✅ Paper updated with detailed bottleneck analysis in Limitations section.

**Session N+4 (DONE — v2 re-validation + sustained/scaling benchmarks):**
1. ✅ v2 `query_memory_fast()` validated: AV flipped from fail to pass.
2. ✅ Sustained benchmarks (15-30s) added and run.
3. ✅ Scaling sweep (1K-50K) added and run.
4. ✅ Paper updated with all results + new Engineering Methodology section.
5. ✅ Results moved to `results/` folder.

**Session N+5 (DONE — v3 device-driven pipeline):**
1. ✅ Device-driven BFS kernel (no inter-hop cudaStreamSynchronize).
2. ✅ GPU-side result compaction kernel (O(N) D2H -> O(visited)).
3. ✅ Pinned host memory for async H2D transfer.
4. ✅ GPU keepalive for low-rate workloads (--keepalive flag).
5. ✅ Paper: new Engineering Methodology section + Industry Adoption Path.

**Session N+6 (next — v3 re-validation):**
1. User runs `make clean && make` on vast.ai A100 → rebuild with v3 pipeline.
2. Run full benchmark suite:
   `make check && make demo-av && make demo-robot && make demo-ar && make demo-voice`
   `make bench-av && make bench-robot && make bench-ar && make bench-voice`
   `make bench-av-keepalive && make bench-sustained && make bench-scale`
3. Update paper tables with v3 measured numbers.
4. Tag as v0.3.0 and push.

**Session N+7 (after v0.3.0):**
1. Run FAISS GPU flat scan and cuVS CAGRA at same corpus sizes (1K-50K)
   for direct recall-vs-latency comparison.
2. Update paper comparison table (Table 7) with measured numbers.
3. Submit paper to arXiv under cs.DC with cs.IR cross-listing.

## Scientific integrity notes

**Do not claim benchmark results the project has not actually run.** The
repo and paper must clearly distinguish:

- **Measured** — numbers produced by running `./validate` on real hardware.
- **Projected** — numbers scaled from CPU reference by per-kernel factors.

All numbers in the current paper and README are **measured** on an NVIDIA
A100X (80 GB, CUDA 12.0) unless explicitly noted otherwise.

## References to existing files

When working with the codebase, key files to read first:
- `include/memory_cuda.cuh` — canonical source for the retrieval API
- `src/memory_cuda.cu` — canonical source for the four kernels
- `src/memory_graph.cpp` — NSN construction
- `src/validate.cu` — template for writing new harness binaries
- `docs/ARCHITECTURE.md` — the deep technical reference

## Author context

This is a research project by Antonello Fratepietro. The paper is
intended for arXiv submission in the cs.DC category (with cs.IR
cross-listing) and potential commercial development. The commercial
target is the bespoke C++ circular buffer market in production
autonomy, robotics, AR/VR, and real-time voice stacks. Teams at Waymo,
Figure, Tesla Optimus, Boston Dynamics, Apple Vision Pro, and dozens of
others currently reinvent this wheel for every new project; a
general-purpose GPU-resident substrate with native cross-modal retrieval
and deadline guarantees is the product.
