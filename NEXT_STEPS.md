# Next Steps

## Current status

The project has a working CUDA retrieval engine with six pipeline versions
across six optimization rounds, validated on four GPUs, with a
same-hardware FAISS/CAGRA comparison and an arXiv paper documenting the
full engineering methodology.

### What's done

- **CUDA engine** — 11 kernels across 6 pipeline versions, all validated
- **Hardware validation** — A100X, A100 PCIE, A100 SXM4, RTX 5060 Ti
- **Six optimization rounds**, each addressing specific bottlenecks
- **Same-hardware FAISS comparison** — GPU kernel time matches FAISS Flat
- **FP16 + CUDA Graph** — scales to 1M memories at 6.5 ms p99
- **CMNG graph ANN** — CAGRA-style graph ANN with cross-modal bridges
- **cuBLAS + CUB** — production-grade primitives, current default
- **arXiv paper** — revised with ablation analysis, failure modes,
  tightened structure, softened claims
- **Engine server** — JSON-line REPL for Python adapters

### What's remaining

| Task | Cost | Time | Notes |
|------|------|------|-------|
| Empirical ablation studies | ~$5 | 2 hours | Disable bridges/hubs/vary BFS depth, measure recall+latency |
| Same-dataset FAISS recall comparison | ~$2 | 1 hour | Match recall@K targets, not just latency |
| arXiv submission | $0 | 10 min | Paper finalized |
| Python bindings (pybind11) | $0 | 1-2 days | Enable LangChain/LlamaIndex integration |
| Multi-GPU sharding | $10+ | 1 week | NVLink cross-shard BFS |

---

## Priority order

### 1. Empirical ablation studies (highest priority)

The paper includes a structural ablation analysis but reviewers will
expect measured numbers. Run the engine with:

- Cross-modal bridges disabled (Phase 5 removed)
- Hub supernodes disabled (Phase 3 removed)
- BFS depth h=1, h=2, h=3
- NSN vs flat brute-force (skip BFS entirely)

Measure recall@10 and wall-clock p99 for each configuration at N=2K, 10K, 50K.

### 2. Recall-matched FAISS comparison

The current comparison matches hardware but not recall targets. Run FAISS
GPU Flat and cuVS CAGRA at the same corpus sizes with the same embeddings,
then compare at equal recall@10 (e.g., 0.85, 0.90, 0.95).

### 3. arXiv submission

```bash
cd paper && tectonic --reruns 3 main.tex
```

Upload to [arxiv.org/submit](https://arxiv.org/submit) under `cs.DC` with
`cs.IR` cross-listing.

### 4. Python bindings

pybind11 wrapper exposing `add()`, `query()`, `delete()`, `stats()` for
integration with LangChain, LlamaIndex, and custom Python pipelines.

---

## Cost summary

| Phase | Cost | Time |
|-------|------|------|
| Ablation studies | ~$5 | 2 hours |
| Recall comparison | ~$2 | 1 hour |
| arXiv submission | $0 | 10 min |
| **Total to submission** | **~$7** | **~3 hours** |
