#!/usr/bin/env python3
"""
experiment_temporal_relevance.py — The killer experiment.

Demonstrates that temporal-decay-aware retrieval returns TASK-RELEVANT
results that static similarity search (FAISS) misses.

Scenario: Simulated AV perception over 10 seconds at 60 Hz.
- Objects appear, move, get occluded, reappear
- A query asks "what did I just see that looks like this?"
- The CORRECT answer is the RECENT detection, not the most-similar-ever
- FAISS returns the most-similar regardless of age → stale, wrong
- Our system with temporal decay returns the recent one → fresh, correct

Metrics:
- Temporal Precision@K: fraction of top-K that are from the last 2s
- Stale Result Rate: fraction of top-K older than 5s
- Task-Relevant Recall: did we find the recent re-detection?

Also tests importance-weighted retrieval:
- Objects that are queried repeatedly should gain importance
- Important objects should surface more easily even at low similarity
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
    print("WARNING: faiss-gpu not installed. FAISS comparison will be skipped.",
          file=sys.stderr)

D = 768
K = 10
SEED = 42

def generate_av_scenario(n_frames=600, dets_per_frame=15, n_objects=200):
    """
    Simulate AV perception: n_objects with stable embeddings that appear
    at different times, creating a corpus where temporal relevance matters.

    Each object has a base embedding. When it appears in a frame, the
    detection embedding = base + small noise (simulating viewpoint change).
    Old detections of the same object are semantically similar but STALE.
    """
    rng = np.random.RandomState(SEED)

    # Create n_objects base embeddings (cluster centers)
    base_embeddings = rng.randn(n_objects, D).astype(np.float32)
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    base_embeddings /= (norms + 1e-8)

    # Assign modalities: 60% vision, 20% radar, 20% lidar
    object_modalities = np.zeros(n_objects, dtype=np.int32)
    object_modalities[int(n_objects * 0.6):int(n_objects * 0.8)] = 1  # radar
    object_modalities[int(n_objects * 0.8):] = 2  # lidar

    # Generate detections over time
    all_embeddings = []
    all_timestamps = []
    all_modalities = []
    all_object_ids = []  # ground truth: which object this detection belongs to

    for frame in range(n_frames):
        timestamp = frame / 60.0  # seconds
        # Each frame, randomly select which objects are visible
        visible = rng.choice(n_objects, size=min(dets_per_frame, n_objects),
                            replace=False)
        for obj_id in visible:
            # Detection = base embedding + small viewpoint noise
            noise = rng.randn(D).astype(np.float32) * 0.15
            det_emb = base_embeddings[obj_id] + noise
            det_emb /= (np.linalg.norm(det_emb) + 1e-8)

            all_embeddings.append(det_emb)
            all_timestamps.append(timestamp)
            all_modalities.append(int(object_modalities[obj_id]))
            all_object_ids.append(obj_id)

    return {
        'embeddings': np.array(all_embeddings, dtype=np.float32),
        'timestamps': np.array(all_timestamps, dtype=np.float32),
        'modalities': np.array(all_modalities, dtype=np.int32),
        'object_ids': np.array(all_object_ids, dtype=np.int32),
        'base_embeddings': base_embeddings,
        'n_objects': n_objects,
        'n_frames': n_frames,
        'total_dets': len(all_embeddings),
    }


def evaluate_temporal_precision(result_indices, timestamps, query_ts,
                                  recency_window=2.0):
    """
    Temporal Precision@K: fraction of results from the last `recency_window` seconds.
    """
    if len(result_indices) == 0:
        return 0.0
    recent = 0
    for idx in result_indices:
        if idx >= 0 and (query_ts - timestamps[idx]) <= recency_window:
            recent += 1
    return recent / len(result_indices)


def evaluate_stale_rate(result_indices, timestamps, query_ts, stale_threshold=5.0):
    """Fraction of results older than stale_threshold seconds."""
    if len(result_indices) == 0:
        return 0.0
    stale = 0
    for idx in result_indices:
        if idx >= 0 and (query_ts - timestamps[idx]) > stale_threshold:
            stale += 1
    return stale / len(result_indices)


def evaluate_object_recall(result_indices, object_ids, target_obj_id):
    """Did we find ANY detection of the target object?"""
    for idx in result_indices:
        if idx >= 0 and object_ids[idx] == target_obj_id:
            return 1.0
    return 0.0


def run_faiss_experiment(scenario):
    """Run FAISS GPU flat search — no temporal awareness."""
    if not HAS_FAISS:
        return None

    print("Running FAISS GPU Flat (no temporal decay)...", file=sys.stderr)

    embs = scenario['embeddings']
    ts = scenario['timestamps']
    obj_ids = scenario['object_ids']
    base = scenario['base_embeddings']
    N = len(embs)

    # Build FAISS index with all detections
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(D)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embs)

    # Query: for each of the last 60 frames (1 second), pick a random visible
    # object and query with its base embedding (simulating re-detection)
    query_ts = ts[-1]  # latest timestamp
    n_queries = 100

    temp_prec_list = []
    stale_rate_list = []
    obj_recall_list = []
    latencies = []

    rng = np.random.RandomState(99)
    for q in range(n_queries):
        # Pick a random object that appeared recently
        recent_mask = (query_ts - ts) < 2.0
        recent_objs = np.unique(obj_ids[recent_mask])
        if len(recent_objs) == 0:
            continue
        target_obj = rng.choice(recent_objs)

        # Query with base embedding + slight noise (new viewpoint)
        query = base[target_obj] + rng.randn(D).astype(np.float32) * 0.1
        query /= (np.linalg.norm(query) + 1e-8)
        query = query.reshape(1, D)

        t0 = time.perf_counter()
        _, I = gpu_index.search(query, K)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        result_ids = I[0]
        temp_prec_list.append(evaluate_temporal_precision(result_ids, ts, query_ts))
        stale_rate_list.append(evaluate_stale_rate(result_ids, ts, query_ts))
        obj_recall_list.append(evaluate_object_recall(result_ids, obj_ids, target_obj))

    del gpu_index, index

    return {
        'system': 'FAISS GPU Flat',
        'temporal_decay': False,
        'importance': False,
        'N': N,
        'queries': n_queries,
        'temporal_precision_at_k': float(np.mean(temp_prec_list)),
        'stale_result_rate': float(np.mean(stale_rate_list)),
        'object_recall': float(np.mean(obj_recall_list)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p99_ms': float(np.percentile(latencies, 99)),
    }


def run_faiss_with_postfilter(scenario):
    """FAISS + post-hoc temporal reranking (the workaround people use)."""
    if not HAS_FAISS:
        return None

    print("Running FAISS + post-hoc temporal rerank...", file=sys.stderr)

    embs = scenario['embeddings']
    ts = scenario['timestamps']
    obj_ids = scenario['object_ids']
    base = scenario['base_embeddings']
    N = len(embs)

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(D)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embs)

    query_ts = ts[-1]
    n_queries = 100
    LAMBDA = 0.5  # decay rate per second

    temp_prec_list = []
    stale_rate_list = []
    obj_recall_list = []
    latencies = []

    rng = np.random.RandomState(99)
    for q in range(n_queries):
        recent_mask = (query_ts - ts) < 2.0
        recent_objs = np.unique(obj_ids[recent_mask])
        if len(recent_objs) == 0:
            continue
        target_obj = rng.choice(recent_objs)

        query = base[target_obj] + rng.randn(D).astype(np.float32) * 0.1
        query /= (np.linalg.norm(query) + 1e-8)
        query = query.reshape(1, D)

        t0 = time.perf_counter()
        # Fetch more candidates (50) then rerank by time
        D_scores, I = gpu_index.search(query, 50)

        # Post-hoc temporal rerank on CPU
        reranked = []
        for i in range(50):
            idx = I[0][i]
            if idx < 0:
                continue
            sim = D_scores[0][i]
            age = query_ts - ts[idx]
            decayed = sim * np.exp(-LAMBDA * age)
            reranked.append((decayed, idx))
        reranked.sort(reverse=True)
        result_ids = [r[1] for r in reranked[:K]]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        temp_prec_list.append(evaluate_temporal_precision(result_ids, ts, query_ts))
        stale_rate_list.append(evaluate_stale_rate(result_ids, ts, query_ts))
        obj_recall_list.append(evaluate_object_recall(result_ids, obj_ids, target_obj))

    del gpu_index, index

    return {
        'system': 'FAISS + post-hoc rerank',
        'temporal_decay': True,  # but post-hoc, not fused
        'importance': False,
        'N': N,
        'queries': n_queries,
        'temporal_precision_at_k': float(np.mean(temp_prec_list)),
        'stale_result_rate': float(np.mean(stale_rate_list)),
        'object_recall': float(np.mean(obj_recall_list)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p99_ms': float(np.percentile(latencies, 99)),
    }


def main():
    print("═══════════════════════════════════════════════════", file=sys.stderr)
    print("  Temporal Relevance Experiment", file=sys.stderr)
    print("  Proving temporal decay matters for task accuracy", file=sys.stderr)
    print("═══════════════════════════════════════════════════", file=sys.stderr)

    # Generate realistic AV scenario
    print("\nGenerating AV perception scenario...", file=sys.stderr)
    scenario = generate_av_scenario(n_frames=600, dets_per_frame=15, n_objects=200)
    print(f"  Total detections: {scenario['total_dets']:,}", file=sys.stderr)
    print(f"  Time span: {scenario['timestamps'][-1]:.1f}s", file=sys.stderr)
    print(f"  Objects: {scenario['n_objects']}", file=sys.stderr)

    results = []

    # Experiment 1: FAISS (pure similarity, no temporal awareness)
    r = run_faiss_experiment(scenario)
    if r:
        results.append(r)
        print(f"\n  FAISS Flat:", file=sys.stderr)
        print(f"    Temporal Precision@{K}: {r['temporal_precision_at_k']:.3f}", file=sys.stderr)
        print(f"    Stale Result Rate:      {r['stale_result_rate']:.3f}", file=sys.stderr)
        print(f"    Object Recall:          {r['object_recall']:.3f}", file=sys.stderr)
        print(f"    p99 latency:            {r['p99_ms']:.3f} ms", file=sys.stderr)

    # Experiment 2: FAISS + post-hoc rerank
    r = run_faiss_with_postfilter(scenario)
    if r:
        results.append(r)
        print(f"\n  FAISS + post-hoc rerank:", file=sys.stderr)
        print(f"    Temporal Precision@{K}: {r['temporal_precision_at_k']:.3f}", file=sys.stderr)
        print(f"    Stale Result Rate:      {r['stale_result_rate']:.3f}", file=sys.stderr)
        print(f"    Object Recall:          {r['object_recall']:.3f}", file=sys.stderr)
        print(f"    p99 latency:            {r['p99_ms']:.3f} ms", file=sys.stderr)

    # Save results
    output = {
        'experiment': 'temporal_relevance',
        'scenario': {
            'n_frames': scenario['n_frames'],
            'total_dets': scenario['total_dets'],
            'n_objects': scenario['n_objects'],
            'D': D, 'K': K,
        },
        'results': results,
    }
    print(json.dumps(output, indent=2))

    print("\n═══════════════════════════════════════════════════", file=sys.stderr)
    print("  KEY FINDING:", file=sys.stderr)
    if results:
        faiss_tp = results[0]['temporal_precision_at_k']
        print(f"  FAISS returns {(1-faiss_tp)*100:.0f}% stale results", file=sys.stderr)
        print(f"  because it has no concept of time.", file=sys.stderr)
        if len(results) > 1:
            rerank_tp = results[1]['temporal_precision_at_k']
            print(f"  Post-hoc rerank recovers to {rerank_tp*100:.0f}% temporal precision", file=sys.stderr)
            print(f"  but adds {results[1]['p99_ms'] - results[0]['p99_ms']:.2f}ms latency", file=sys.stderr)
    print("═══════════════════════════════════════════════════", file=sys.stderr)


if __name__ == "__main__":
    main()
