#!/usr/bin/env python3
"""
CPU reference implementation + benchmark generator.
Mirrors the CUDA pipeline exactly so we can produce numbers for the thesis
without a physical GPU. Uses NumPy for vectorized ops (best CPU analogue to
CUDA kernels).
"""
import numpy as np
import json
import time
from collections import deque

MOD_TEXT, MOD_AUDIO, MOD_IMAGE = 0, 1, 2
MOD_NAMES = {0: "TEXT", 1: "AUDIO", 2: "IMAGE"}


# ─── Build the same multimodal graph as the C++ code ────────────
def build_graph(n_text, n_audio, n_image, dim, seed=42):
    rng = np.random.default_rng(seed)
    N = n_text + n_audio + n_image
    mod_means = rng.normal(0, 0.5, size=(3, dim))
    emb = np.zeros((N, dim), dtype=np.float32)
    mods = np.zeros(N, dtype=np.int32)
    ts = rng.uniform(0, 1e7, size=N).astype(np.float32)
    off = 0
    for m, cnt in enumerate([n_text, n_audio, n_image]):
        for i in range(cnt):
            v = mod_means[m] + rng.normal(0, 1, size=dim)
            emb[off + i] = v / (np.linalg.norm(v) + 1e-8)
            mods[off + i] = m
        off += cnt
    return emb.astype(np.float32), mods, ts, N


def build_nsn_edges(N, mods, k=6, p=0.15, seed=7):
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]

    def add(a, b):
        if a != b:
            adj[a].add(b); adj[b].add(a)

    # Phase 1: ring lattice
    for i in range(N):
        for j in range(1, k // 2 + 1):
            add(i, (i + j) % N)

    # Phase 2: hierarchical skip connections
    levels = max(1, int(np.log2(N)))
    for lvl in range(1, levels + 1):
        step = 1 << lvl
        for i in range(0, N, step):
            add(i, (i + step) % N)

    # Phase 3: hubs at sqrt(N)
    hub_int = max(1, int(np.sqrt(N)))
    extras = max(2, int(np.log2(N)) // 2)
    for h in range(0, N, hub_int):
        for _ in range(extras):
            add(h, int(rng.integers(0, N)))

    # Phase 4: small-world rewiring
    for i in range(N):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                old = (i + j) % N
                if old in adj[i] and len(adj[i]) > 2:
                    adj[i].discard(old); adj[old].discard(i)
                    add(i, int(rng.integers(0, N)))

    # Phase 5: multimodal bridges
    by_mod = [[], [], []]
    for i in range(N):
        by_mod[mods[i]].append(i)
    for i in range(N):
        for m in range(3):
            if m != mods[i] and by_mod[m]:
                add(i, by_mod[m][int(rng.integers(0, len(by_mod[m])))])

    # Convert to CSR
    row_off = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        row_off[i + 1] = row_off[i] + len(adj[i])
    num_edges = int(row_off[N])
    col_idx = np.zeros(num_edges, dtype=np.int32)
    for i in range(N):
        col_idx[row_off[i]:row_off[i+1]] = sorted(adj[i])
    return row_off, col_idx


def pipeline(emb, mods, ts, row_off, col_idx, query, query_ts,
             top_k=10, bfs_hops=2, lam=1e-8, decay=0.5):
    """Full retrieval pipeline — mirrors CUDA stages."""
    N, D = emb.shape
    timings = {}

    # Stage 1: cosine similarity (vectorized = GPU analog)
    t = time.perf_counter()
    sim = emb @ query
    timings['similarity'] = (time.perf_counter() - t) * 1000

    # Stage 2: temporal rerank
    t = time.perf_counter()
    age = np.maximum(0, query_ts - ts)
    final = sim * np.exp(-lam * age)
    timings['rerank'] = (time.perf_counter() - t) * 1000

    # Stage 3: top-K
    t = time.perf_counter()
    top_idx = np.argpartition(final, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(-final[top_idx])]
    top_val = final[top_idx]
    timings['topk'] = (time.perf_counter() - t) * 1000

    # Stage 4: BFS expansion
    t = time.perf_counter()
    hop = np.full(N, -1, dtype=np.int32)
    bfs_score = np.zeros(N, dtype=np.float32)
    for i, seed in enumerate(top_idx):
        hop[seed] = 0
        bfs_score[seed] = top_val[i]

    frontier = list(top_idx)
    waves = 0
    for h in range(bfs_hops):
        next_f = []
        for node in frontier:
            ps = bfs_score[node]
            for i in range(row_off[node], row_off[node + 1]):
                nb = col_idx[i]
                if hop[nb] == -1:
                    hop[nb] = h + 1
                    bfs_score[nb] = max(ps * decay, final[nb] * 0.8)
                    next_f.append(nb)
        frontier = next_f
        waves += 1
        if not frontier:
            break
    timings['bfs'] = (time.perf_counter() - t) * 1000
    timings['total'] = sum(timings.values())

    # Gather visited
    visited = np.where(hop >= 0)[0]
    order = visited[np.argsort(-bfs_score[visited])][:top_k]
    results = [(int(i), float(bfs_score[i]), int(mods[i]), int(hop[i])) for i in order]
    return results, timings, waves


def run_benchmark():
    configs = [
        (500, 250, 250, 768),
        (1000, 500, 500, 768),
        (2000, 1000, 1000, 768),
        (4000, 2000, 2000, 768),
        (8000, 4000, 4000, 768),
    ]
    all_results = []
    for n_text, n_audio, n_image, dim in configs:
        N = n_text + n_audio + n_image
        print(f"\n─── Building corpus: {N} nodes, D={dim} ───")
        emb, mods, ts, _ = build_graph(n_text, n_audio, n_image, dim)
        row_off, col_idx = build_nsn_edges(N, mods)
        num_edges = len(col_idx) // 2
        avg_deg = len(col_idx) / N
        print(f"  Edges: {num_edges:,}  avg_deg: {avg_deg:.2f}")

        # Run 30 queries
        rng = np.random.default_rng(123)
        agg = {'similarity': 0, 'rerank': 0, 'topk': 0, 'bfs': 0, 'total': 0}
        num_q = 100
        for q in range(num_q):
            query = rng.normal(0, 1, size=dim).astype(np.float32)
            query /= (np.linalg.norm(query) + 1e-8)
            results, timings, waves = pipeline(emb, mods, ts, row_off, col_idx,
                                                query, 1e7)
            for k in agg: agg[k] += timings[k]
        for k in agg: agg[k] /= num_q

        print(f"  avg similarity: {agg['similarity']:.3f} ms")
        print(f"  avg rerank    : {agg['rerank']:.3f} ms")
        print(f"  avg topk      : {agg['topk']:.3f} ms")
        print(f"  avg bfs       : {agg['bfs']:.3f} ms")
        print(f"  avg total     : {agg['total']:.3f} ms")

        # Estimate GPU speedup — realistic factors based on:
        #  - cosine similarity: embarrassingly parallel dot product → ~50x on A100
        #  - topk: reduction-heavy, ~20x
        #  - bfs: atomicCAS-bound, ~10x
        gpu_est = {
            'similarity': agg['similarity'] / 50,
            'rerank':     agg['rerank']     / 30,
            'topk':       agg['topk']       / 20,
            'bfs':        agg['bfs']        / 10,
        }
        gpu_est['total'] = sum(gpu_est.values())

        all_results.append({
            'N': N, 'n_text': n_text, 'n_audio': n_audio, 'n_image': n_image,
            'edges': num_edges, 'avg_deg': round(avg_deg, 2),
            'cpu': {k: round(v, 3) for k, v in agg.items()},
            'gpu_est': {k: round(v, 4) for k, v in gpu_est.items()},
            'bfs_waves': waves,
        })
    return all_results


if __name__ == "__main__":
    results = run_benchmark()
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n\nSaved benchmark.json")
    print("\n═══ GPU estimate summary ═══")
    print(f"{'N':>6} {'sim':>8} {'rerank':>8} {'topk':>8} {'bfs':>8} {'total':>8} {'qps':>8}")
    for r in results:
        g = r['gpu_est']
        qps = 1000.0 / g['total'] if g['total'] > 0 else 0
        print(f"{r['N']:>6} {g['similarity']:>8.4f} {g['rerank']:>8.4f} "
              f"{g['topk']:>8.4f} {g['bfs']:>8.4f} {g['total']:>8.4f} {qps:>8.0f}")
