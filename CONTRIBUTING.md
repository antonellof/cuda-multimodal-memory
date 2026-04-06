# Contributing to MARS

Thanks for your interest! This project welcomes contributions across several
areas.

## What we're looking for

### High priority

- **Hardware benchmark runs on new GPUs.** The project is validated on A100
  PCIE, A100 SXM4, A100X, and RTX 5060 Ti. If you have access to an H100,
  RTX 4090, Jetson Orin, or other compute-capability-7.0+ GPU, run
  `make && make check && make bench-mars` and open an issue with your results.
- **Multi-GPU sharding.** The current design is single-GPU. Extending to NVLink
  / NVSwitch setups with cross-shard BFS would unlock larger corpuses.
- **Empirical ablation studies.** Measuring recall and latency with individual
  NSN construction phases disabled (e.g., without cross-modal bridges, without
  hubs, at different BFS depths) to quantify each phase's contribution.

### Also welcome

- Bug reports with minimal reproductions
- Documentation improvements
- Alternative memory graph topologies (as long as they preserve the cross-modal
  bridge property — every node must retain an edge to every other modality)
- Python bindings (pybind11) so the engine is callable from LangChain / LlamaIndex

### Not currently in scope

- Fact extraction from conversations (belongs in a higher-level layer)
- User profile assembly (belongs in a higher-level layer)
- Connector ecosystems for Gmail / Drive / Notion (belongs in a higher-level layer)
- LLM integration (out of scope — this is the retrieval substrate only)

## Development workflow

1. **Fork** the repo on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your change: `git checkout -b feature/my-change`
4. **Build and test** locally: `make && ./memory_engine`
5. **Commit** with a clear message
6. **Push** to your fork and open a **pull request**

## Code style

- **C++17** for host code, **CUDA C++** for device code
- 4-space indentation, no tabs
- Braces on same line for functions, kernels, and control flow
- `snake_case` for variables and functions, `CamelCase` for types
- Prefer `int32_t` / `float` explicit types over `int` / `double`
- Every kernel gets a comment block explaining its grid/block layout
- `__restrict__` on all pointer arguments to kernels
- `CUDA_CHECK()` on every runtime API call (see `include/memory_cuda.cuh`)

## Testing a change

The project has host-only unit tests (7/7 passing) and GPU validation:

```bash
# Host-only tests (no GPU needed)
make tests

# GPU validation (requires NVIDIA GPU)
make && make check

# Full benchmark suite
make bench-mars                # MARS pipeline (cuBLAS + CUB)
make bench-av && make bench-robot && make bench-ar && make bench-voice

# Memory checking
compute-sanitizer ./memory_engine 1000 500 500 768
```

## Reporting bugs

Please open an issue with:

- A minimal reproduction
- Your GPU model, CUDA version, and compiler version
- The exact command that triggered the bug
- Any error output or unexpected results

## Questions

For general questions about the architecture or how to use the engine, please
open a **Discussion** rather than an issue.
