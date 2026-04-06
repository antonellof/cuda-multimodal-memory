# Benchmarks

Detailed performance data for MARS.
All numbers are measured on real hardware via vast.ai.

---

## MARS — Current Pipeline

MARS replaces the custom similarity kernel with **cuBLAS SGEMV** and the
serial top-K with **CUB DeviceRadixSort**, eliminating the top-K
bottleneck that was 90% of pipeline time.

### Same-hardware comparison (A100 SXM4 80GB, D=768, K=10)

All systems measured on the same machine with the same data:

| System | N=2.4K | N=10K | N=20K | N=50K | Cross-modal | Temporal decay |
|--------|--------|-------|-------|-------|-------------|----------------|
| FAISS GPU Flat | 0.10 ms | 0.12 ms | 0.18 ms | 0.35 ms | No | No |
| FAISS GPU IVF | 0.13 ms | 0.15 ms | 0.30 ms | 0.28 ms | No | No |
| cuVS CAGRA | 2.60 ms | 2.29 ms | 2.29 ms | 2.47 ms | No | No |
| **MARS** | **0.26 ms** | **0.34 ms** | **0.36 ms** | **0.44 ms** | **Yes** | **Yes** |

### GPU kernel time (excluding host overhead)

| System | N=2.4K | N=10K | N=50K |
|--------|--------|-------|-------|
| FAISS Flat | 0.09 ms | 0.12 ms | 0.35 ms |
| **MARS** | **0.10 ms** | **0.17 ms** | **0.29 ms** |

---

## Demonstrator results

### A100 PCIE 40GB (all pass)

| Bench | Rate | Corpus | Budget | Wall p99 | GPU p99 | Miss rate | Verdict |
|-------|------|--------|--------|----------|---------|-----------|---------|
| AV | 60 Hz | 2,400 | 1 ms | 0.87 ms | 0.68 ms | 0% | **pass** |
| Robot | 1 kHz | 6,000 | 1 ms | 0.76 ms | 0.69 ms | 0% | **pass** |
| AR/VR | 90 Hz | 20,000 | 5 ms | 1.56 ms | 1.37 ms | 0% | **pass** |
| Voice | 30 Hz | 3,000 | 20 ms | 0.88 ms | 0.68 ms | 0% | **pass** |

### RTX 5060 Ti 16GB (all pass, 2.3–2.8× faster)

| Bench | Rate | Corpus | Budget | Wall p99 | GPU p99 | Verdict |
|-------|------|--------|--------|----------|---------|---------|
| AV | 60 Hz | 2,400 | 1 ms | **0.31 ms** | 0.23 ms | **pass** |
| Robot | 1 kHz | 6,000 | 1 ms | **0.29 ms** | 0.23 ms | **pass** |
| AR/VR | 90 Hz | 20,000 | 5 ms | **0.67 ms** | 0.58 ms | **pass** |
| Voice | 30 Hz | 3,000 | 20 ms | **0.32 ms** | 0.23 ms | **pass** |

---

## Sustained benchmarks (30s real-world duration)

### A100 PCIE (all pass)

| Test | Rate | Duration | Corpus | Budget | p99 wall | Misses | Verdict |
|------|------|----------|--------|--------|----------|--------|---------|
| AV 30s | 60 Hz | 30s | 5K | 1 ms | 0.91 ms | 0 | **pass** |
| Robot 15s | 1 kHz | 15s | 10K | 1 ms | 0.66 ms | 0 | **pass** |
| AR 30s | 90 Hz | 30s | 50K | 5 ms | 1.70 ms | 0 | **pass** |
| Voice 30s | 30 Hz | 30s | 10K | 20 ms | 0.96 ms | 0 | **pass** |

### RTX 5060 Ti (all pass)

| Test | Rate | Duration | Corpus | Budget | p99 wall | Misses | Verdict |
|------|------|----------|--------|--------|----------|--------|---------|
| AV 30s | 60 Hz | 30s | 5K | 1 ms | 0.32 ms | 0 | **pass** |
| Robot 15s | 1 kHz | 15s | 10K | 1 ms | 0.32 ms | 0 | **pass** |
| AR 30s | 90 Hz | 30s | 50K | 5 ms | 0.91 ms | 0 | **pass** |
| Voice 30s | 30 Hz | 30s | 10K | 20 ms | 0.36 ms | 0 | **pass** |

---

## Scaling sweeps

### FP32 (60 Hz × 15s, 1 ms budget)

**A100 PCIE:**

| N | Wall p99 | GPU p99 | Verdict |
|---|---------|---------|---------|
| 1K | 0.84 ms | 0.67 ms | **pass** |
| 5K | 0.91 ms | 0.70 ms | **pass** |
| 10K | 0.96 ms | 0.75 ms | **pass** |
| 20K | 1.59 ms | 1.38 ms | fail |
| 50K | 1.69 ms | 1.51 ms | fail |

**RTX 5060 Ti** (all pass):

| N | Wall p99 | GPU p99 | Verdict |
|---|---------|---------|---------|
| 1K | 0.29 ms | 0.22 ms | **pass** |
| 5K | 0.32 ms | 0.24 ms | **pass** |
| 10K | 0.35 ms | 0.27 ms | **pass** |
| 20K | 0.66 ms | 0.58 ms | **pass** |
| 50K | 0.90 ms | 0.82 ms | **pass** |

### FP16 + CUDA Graph (large corpus, 60 Hz)

| N | RTX 5060 Ti | A100 SXM4 |
|---|------------|-----------|
| 50K | 0.89 ms | 1.22 ms |
| 100K | 1.18 ms | 1.38 ms |
| 200K | 1.75 ms | 1.67 ms |
| 500K | 3.53 ms | 2.51 ms |
| 1M | 6.51 ms | 7.70 ms |

---

## Optimization history

Six pipeline versions, each addressing a specific bottleneck:

| Version | Key change | Impact |
|---------|-----------|--------|
| Baseline | Per-query cudaMalloc, sync D2H | AV=1.41ms, Robot=1.12ms (both fail) |
| Targeted fixes | Pre-allocated buffers, GPU BFS init | AV=0.96ms (pass), Robot=1.21ms (fail) |
| Device-driven | Device-driven BFS, result compaction, pinned memory | AV=0.87ms, Robot=0.76ms (both pass) |
| FP16+Graph | FP16 similarity, CUDA Graph capture | 1M memories at 6.5ms |
| CMNG | CAGRA-style graph ANN + cross-modal bridges | Nearly flat latency 50K→200K |
| **cuBLAS+CUB (current)** | cuBLAS SGEMV similarity, CUB radix top-K | **GPU kernel matches FAISS Flat** |

### Current vs baseline improvement

| N | Baseline p99 | MARS p99 | Speedup |
|---|------------|------------|---------|
| 2.4K | 0.87 ms | **0.26 ms** | 3.3× |
| 10K | 0.96 ms | **0.34 ms** | 2.8× |
| 20K | 1.59 ms | **0.36 ms** | 4.4× |
| 50K | 1.69 ms | **0.44 ms** | 3.8× |

---

## Running benchmarks

```bash
# MARS pipeline (recommended)
make bench-mars                    # cuBLAS + CUB at N=2.4K, 10K, 20K, 50K

# Additional pipelines
make bench-av                    # 60 Hz, 2.4K memories
make bench-robot                 # 1 kHz, 6K memories
make bench-ar                    # 90 Hz, 20K memories
make bench-voice                 # 30 Hz, 3K memories
make bench-sustained             # all four 15-30s tests
make bench-scale                 # N = 1K through 50K
make bench-large                 # N = 50K through 1M (FP16+Graph)
make bench-fp16                  # FP32 vs FP16 A/B comparison

# FAISS GPU comparison (requires pip install faiss-gpu-cu12 numpy)
python scripts/bench_faiss_comparison.py

# CMNG graph ANN (experimental)
make bench-cmng                  # N = 10K through 200K
```

---

## GPUs tested

| GPU | VRAM | Architecture | CUDA | Results folder |
|-----|------|-------------|------|---------------|
| A100X 80GB | 80 GB | Ampere sm_80 | 12.0 | `results/a100x-80gb-v2/` |
| A100 PCIE 40GB | 40 GB | Ampere sm_80 | 12.8 | `results/a100-pcie-40gb-v3/` |
| A100 SXM4 80GB | 80 GB | Ampere sm_80 | 12.6–12.8 | `results/a100-sxm4-*` |
| RTX 5060 Ti 16GB | 16 GB | Blackwell sm_120 | 13.0 | `results/rtx5060ti*/` |
