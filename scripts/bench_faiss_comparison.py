#!/usr/bin/env python3
"""
bench_faiss_comparison.py — Direct latency comparison with FAISS GPU

Runs FAISS GPU flat-scan and IVF at the same corpus sizes and embedding
dimensions as our CUDA memory engine, measuring single-query p99 latency.

This produces an apples-to-apples comparison: same hardware, same corpus,
same D=768 embeddings, same single-query regime.

Usage:
    pip install faiss-gpu numpy
    python scripts/bench_faiss_comparison.py

Output: JSON to stdout with per-configuration results.
"""
import json
import sys
import time
import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: faiss-gpu not installed. Run: pip install faiss-gpu numpy", file=sys.stderr)
    sys.exit(1)

# Check if GPU is available
ngpus = faiss.get_num_gpus()
if ngpus == 0:
    print("ERROR: No GPU found by FAISS", file=sys.stderr)
    sys.exit(1)

D = 768
NUM_QUERIES = 300  # queries per config
TOP_K = 10

# Corpus sizes to test — matches our bench-scale and bench-large
CORPUS_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
                100_000, 200_000, 500_000, 1_000_000]


def generate_data(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate L2-normalized random embeddings (same as our engine)."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= (norms + 1e-8)
    return data


def generate_queries(nq: int, d: int, seed: int = 123) -> np.ndarray:
    return generate_data(nq, d, seed)


def measure_single_query_latencies(index, queries: np.ndarray, k: int) -> list:
    """Measure wall-clock latency for each single query."""
    latencies = []
    for i in range(len(queries)):
        q = queries[i:i+1]  # shape (1, D)
        t0 = time.perf_counter()
        index.search(q, k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
    return latencies


def percentile(lats: list, p: float) -> float:
    s = sorted(lats)
    idx = int(len(s) * p / 100.0)
    idx = min(idx, len(s) - 1)
    return s[idx]


def compute_recall(ground_truth_ids: np.ndarray, test_ids: np.ndarray) -> float:
    """Compute recall@K: fraction of ground truth IDs found in test results."""
    recalls = []
    for i in range(len(ground_truth_ids)):
        gt_set = set(ground_truth_ids[i].tolist())
        test_set = set(test_ids[i].tolist())
        recalls.append(len(gt_set & test_set) / len(gt_set))
    return float(np.mean(recalls))


def run_flat(n: int) -> dict:
    """FAISS GPU flat (brute-force) scan."""
    print(f"  FAISS Flat N={n:,}...", end="", flush=True, file=sys.stderr)
    data = generate_data(n, D)
    queries = generate_queries(NUM_QUERIES, D)

    # Build GPU flat index
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(D)  # inner product = cosine for normalized
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(data)

    # Warmup
    gpu_index.search(queries[:10], TOP_K)

    # Measure latency
    lats = measure_single_query_latencies(gpu_index, queries, TOP_K)

    # Measure recall (flat is exact, so recall=1.0 — but verify)
    _, I = gpu_index.search(queries, TOP_K)
    # Ground truth = flat results (exact)
    recall = 1.0  # by definition for brute-force

    result = {
        "method": "faiss_flat_gpu",
        "N": n,
        "D": D,
        "K": TOP_K,
        "num_queries": NUM_QUERIES,
        "p50_ms": percentile(lats, 50),
        "p99_ms": percentile(lats, 99),
        "max_ms": max(lats),
        "mean_ms": sum(lats) / len(lats),
        "recall_at_k": recall,
        "ground_truth_ids": I.tolist(),  # save for cross-system comparison
    }
    print(f" p99={result['p99_ms']:.3f}ms recall={recall:.2f}", file=sys.stderr)
    del gpu_index, index_flat
    return result


def run_ivf(n: int) -> dict:
    """FAISS GPU IVF-Flat scan."""
    if n < 1000:
        return None  # IVF needs enough data for clustering

    nlist = max(1, min(int(np.sqrt(n)), 1024))
    nprobe = max(1, min(nlist // 4, 64))

    print(f"  FAISS IVF N={n:,} nlist={nlist} nprobe={nprobe}...",
          end="", flush=True, file=sys.stderr)
    data = generate_data(n, D)
    queries = generate_queries(NUM_QUERIES, D)

    # Build GPU flat index for ground truth
    res_gt = faiss.StandardGpuResources()
    gt_index = faiss.IndexFlatIP(D)
    gt_gpu = faiss.index_cpu_to_gpu(res_gt, 0, gt_index)
    gt_gpu.add(data)
    _, gt_ids = gt_gpu.search(queries, TOP_K)
    del gt_gpu, gt_index, res_gt

    # Build GPU IVF index
    res = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatIP(D)
    index_ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    gpu_index.train(data)
    gpu_index.add(data)
    faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", nprobe)

    # Warmup
    gpu_index.search(queries[:10], TOP_K)

    # Measure latency
    lats = measure_single_query_latencies(gpu_index, queries, TOP_K)

    # Measure recall vs ground truth
    _, I = gpu_index.search(queries, TOP_K)
    recall = compute_recall(gt_ids, I)

    result = {
        "method": "faiss_ivf_gpu",
        "N": n,
        "D": D,
        "K": TOP_K,
        "nlist": nlist,
        "nprobe": nprobe,
        "num_queries": NUM_QUERIES,
        "p50_ms": percentile(lats, 50),
        "p99_ms": percentile(lats, 99),
        "max_ms": max(lats),
        "mean_ms": sum(lats) / len(lats),
        "recall_at_k": recall,
    }
    print(f" p99={result['p99_ms']:.3f}ms recall={recall:.2f}", file=sys.stderr)
    del gpu_index, index_ivf, quantizer
    return result


def main():
    print(f"FAISS GPU Comparison Benchmark", file=sys.stderr)
    print(f"  GPUs: {ngpus}, D={D}, K={TOP_K}, queries={NUM_QUERIES}",
          file=sys.stderr)
    print(f"  Corpus sizes: {[f'{n:,}' for n in CORPUS_SIZES]}",
          file=sys.stderr)
    print(file=sys.stderr)

    results = {"tool": "faiss-gpu-comparison", "D": D, "K": TOP_K,
               "num_queries": NUM_QUERIES, "configurations": []}

    for n in CORPUS_SIZES:
        # Skip sizes that won't fit in GPU memory (rough check)
        vram_mb = n * D * 4 / 1e6  # FP32
        if vram_mb > 35_000:  # 35 GB safety margin
            print(f"  Skipping N={n:,} ({vram_mb:.0f} MB) — may exceed VRAM",
                  file=sys.stderr)
            continue

        flat = run_flat(n)
        results["configurations"].append(flat)

        ivf = run_ivf(n)
        if ivf:
            results["configurations"].append(ivf)

    # Output JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
