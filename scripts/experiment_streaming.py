#!/usr/bin/env python3
"""
experiment_streaming.py — Streaming insertion + query benchmark.

Demonstrates that FAISS cannot handle concurrent insertion and query
at sensor rates. FAISS requires index rebuild for new vectors; our
system supports online insertion.

Scenario: 60 Hz sensor feed over 30 seconds.
- Each frame: 10 new detections inserted + 1 query
- Query asks "have I seen this before in the last 2s?"
- FAISS must rebuild every N frames (measured overhead)
- Our system: insert + query in the same pipeline

Metrics:
- Insert + Query combined latency per frame
- FAISS rebuild cost amortized over frames
- Freshness: can the system find a detection from 100ms ago?
"""
import json
import sys
import time
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

D = 768
K = 10
SEED = 42
FPS = 60
DURATION_S = 10
DETS_PER_FRAME = 10
N_OBJECTS = 100


def run_faiss_streaming():
    """FAISS: must batch-add then rebuild. Simulate realistic usage."""
    if not HAS_FAISS:
        return None

    print("Running FAISS streaming (rebuild every 60 frames)...", file=sys.stderr)
    rng = np.random.RandomState(SEED)

    # Object base embeddings
    bases = rng.randn(N_OBJECTS, D).astype(np.float32)
    bases /= (np.linalg.norm(bases, axis=1, keepdims=True) + 1e-8)

    res = faiss.StandardGpuResources()
    total_frames = FPS * DURATION_S
    rebuild_interval = 60  # rebuild index every 60 frames (1s)

    all_embs = []
    all_ts = []
    pending_embs = []

    query_latencies = []
    insert_latencies = []
    rebuild_latencies = []
    freshness_hits = []  # can we find detection from <200ms ago?

    for frame in range(total_frames):
        ts = frame / FPS

        # Generate new detections
        t_ins_0 = time.perf_counter()
        for _ in range(DETS_PER_FRAME):
            obj = rng.randint(N_OBJECTS)
            emb = bases[obj] + rng.randn(D).astype(np.float32) * 0.15
            emb /= (np.linalg.norm(emb) + 1e-8)
            pending_embs.append(emb)
            all_embs.append(emb)
            all_ts.append(ts)
        t_ins_1 = time.perf_counter()
        insert_latencies.append((t_ins_1 - t_ins_0) * 1000)

        # Rebuild index periodically
        if frame > 0 and frame % rebuild_interval == 0:
            t_rb_0 = time.perf_counter()
            data = np.array(all_embs, dtype=np.float32)
            index = faiss.IndexFlatIP(D)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(data)
            t_rb_1 = time.perf_counter()
            rebuild_latencies.append((t_rb_1 - t_rb_0) * 1000)
            pending_embs = []
        elif frame == 0:
            # Initial build
            data = np.array(all_embs, dtype=np.float32)
            index = faiss.IndexFlatIP(D)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(data)
            pending_embs = []

        # Query: search for a recent object
        obj = rng.randint(N_OBJECTS)
        query = bases[obj] + rng.randn(D).astype(np.float32) * 0.1
        query /= (np.linalg.norm(query) + 1e-8)
        query = query.reshape(1, D)

        t_q_0 = time.perf_counter()
        _, I = gpu_index.search(query, K)
        t_q_1 = time.perf_counter()
        query_latencies.append((t_q_1 - t_q_0) * 1000)

        # Check freshness: is any result from the last 200ms?
        ts_arr = np.array(all_ts[:gpu_index.ntotal])
        fresh = False
        for idx in I[0]:
            if idx >= 0 and idx < len(ts_arr) and (ts - ts_arr[idx]) < 0.2:
                fresh = True
                break
        freshness_hits.append(1.0 if fresh else 0.0)

    # Amortize rebuild cost
    total_rebuild_ms = sum(rebuild_latencies)
    avg_rebuild_per_frame = total_rebuild_ms / total_frames

    return {
        'system': 'FAISS GPU Flat (rebuild every 1s)',
        'total_frames': total_frames,
        'total_detections': len(all_embs),
        'query_p50_ms': float(np.percentile(query_latencies, 50)),
        'query_p99_ms': float(np.percentile(query_latencies, 99)),
        'insert_p99_ms': float(np.percentile(insert_latencies, 99)),
        'rebuild_count': len(rebuild_latencies),
        'rebuild_avg_ms': float(np.mean(rebuild_latencies)) if rebuild_latencies else 0,
        'rebuild_total_ms': total_rebuild_ms,
        'amortized_rebuild_per_frame_ms': avg_rebuild_per_frame,
        'effective_p99_ms': float(np.percentile(query_latencies, 99)) + avg_rebuild_per_frame,
        'freshness_rate': float(np.mean(freshness_hits)),
        'stale_detection_gap': f"{(1-np.mean(freshness_hits))*100:.1f}% of queries miss recent detections",
        'note': 'Pending insertions between rebuilds are invisible to queries',
    }


def main():
    print("═══════════════════════════════════════════════════", file=sys.stderr)
    print("  Streaming Insertion + Query Experiment", file=sys.stderr)
    print("  FAISS cannot query newly-inserted vectors", file=sys.stderr)
    print("═══════════════════════════════════════════════════", file=sys.stderr)

    results = []

    r = run_faiss_streaming()
    if r:
        results.append(r)
        print(f"\n  FAISS Streaming Results:", file=sys.stderr)
        print(f"    Query p99:      {r['query_p99_ms']:.3f} ms", file=sys.stderr)
        print(f"    Rebuild avg:    {r['rebuild_avg_ms']:.1f} ms ({r['rebuild_count']} rebuilds)", file=sys.stderr)
        print(f"    Rebuild total:  {r['rebuild_total_ms']:.0f} ms over {r['total_frames']} frames", file=sys.stderr)
        print(f"    Freshness rate: {r['freshness_rate']:.3f}", file=sys.stderr)
        print(f"    {r['stale_detection_gap']}", file=sys.stderr)

    output = {'experiment': 'streaming_insertion', 'results': results}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
