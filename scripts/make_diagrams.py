#!/usr/bin/env python3
"""
Generates publication-quality diagrams for the paper:
  diag_pipeline.png          — full retrieval pipeline architecture
  diag_graph.png             — multimodal NSN graph with cross-modal bridges
  diag_kernels.png           — CUDA kernel dataflow & warp organization
  diag_benchmark.png         — per-stage timing chart (measured GPU)
  diag_scaling.png           — latency vs corpus size with budget lines
  diag_sustained.png         — sustained vs short-run benchmark comparison
  diag_wallclock.png         — wall-clock vs GPU kernel overhead breakdown
  diag_opt_history.png       — optimization progression
  diag_large_scaling.png     — large corpus scaling (50K-1M)
  diag_fp16_comparison.png   — FP32 vs FP16 vs FP16+Graph
  diag_faiss_comparison.png  — FAISS/CAGRA vs MARS comparison
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 180,
})

# ─── Color palette ──────────────────────────────────────────────
C_TEXT  = "#3498db"   # blue
C_AUDIO = "#e67e22"   # orange
C_IMAGE = "#2ecc71"   # green
C_HUB   = "#9b59b6"   # purple
C_GPU   = "#1a1a2e"   # near-black
C_ACCENT = "#e74c3c"  # red
C_BG    = "#f8f9fa"

# ════════════════════════════════════════════════════════════════
#  DIAGRAM 1: FULL PIPELINE ARCHITECTURE
# ════════════════════════════════════════════════════════════════
def make_pipeline():
    fig, ax = plt.subplots(figsize=(15.5, 8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor("white")

    # ── Title ──
    ax.text(7.75, 7.6, "MARS — Retrieval Pipeline",
            fontsize=15, fontweight="bold", ha="center", color=C_GPU)

    # ── Input modality encoders (left) ──
    encoders = [
        (0.3, 5.8, "Text\n(BERT / E5)",   C_TEXT),
        (0.3, 4.3, "Audio\n(CLAP / Whisper)", C_AUDIO),
        (0.3, 2.8, "Image\n(CLIP / SigLIP)",  C_IMAGE),
    ]
    for x, y, label, color in encoders:
        box = FancyBboxPatch((x, y), 1.8, 1.0,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=color,
                              linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 0.5, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    # Unified embedding space
    ax.add_patch(FancyBboxPatch((2.5, 3.3), 1.6, 2.0,
                  boxstyle="round,pad=0.08",
                  facecolor="#ecf0f1", edgecolor=C_GPU, linewidth=2))
    ax.text(3.3, 4.6, "Unified\n768-D\nEmbedding\nSpace",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_GPU)

    # Arrows encoders → embedding space (all land on the box's vertical span)
    embed_mid_y = 4.3  # vertical center of embedding box
    embed_top   = 5.3
    embed_bot   = 3.3
    for _, y, _, color in encoders:
        src_y = y + 0.5
        # Clamp destination to the embedding box's vertical extent
        dst_y = max(embed_bot + 0.1, min(embed_top - 0.1, src_y))
        ax.annotate("", xy=(2.5, dst_y), xytext=(2.1, src_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # ── GPU boundary ──
    gpu_box = FancyBboxPatch((4.5, 0.6), 8.8, 6.4,
                              boxstyle="round,pad=0.1",
                              facecolor="#fafbfc", edgecolor=C_GPU,
                              linewidth=2.5, linestyle="--")
    ax.add_patch(gpu_box)
    ax.text(12.7, 6.75, "GPU (CUDA)", ha="right", va="center",
            fontsize=10, fontweight="bold", color=C_GPU,
            style="italic")

    # ── Four kernel stages (inside GPU box) ──
    stages = [
        (4.8, 5.2, "Stage 1\ncuBLAS SGEMV\nCosine Similarity\n(matrix-vector multiply)", "#2980b9"),
        (7.0, 5.2, "Stage 2\nTemporal Rerank\nscore × exp(−λ·age)",                       "#16a085"),
        (9.2, 5.2, "Stage 3\nCUB Radix Sort\nTop-K Selection",                          "#8e44ad"),
        (11.4, 5.2, "Stage 4\nBFS Expansion\nwarp-cooperative\n+ atomicCAS",             "#c0392b"),
    ]
    for i, (x, y, label, color) in enumerate(stages):
        box = FancyBboxPatch((x, y - 0.8), 1.8, 1.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=color,
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 0.1, label, ha="center", va="center",
                fontsize=7.3, fontweight="bold", color="white")
        if i < 3:
            ax.annotate("", xy=(stages[i+1][0], y + 0.1),
                        xytext=(x + 1.8, y + 0.1),
                        arrowprops=dict(arrowstyle="->", color=C_GPU, lw=1.8))

    # From embedding space into first kernel
    ax.annotate("", xy=(4.8, 5.3), xytext=(4.1, 4.6),
                arrowprops=dict(arrowstyle="->", color=C_GPU, lw=2))
    ax.text(4.4, 4.9, "query\nvector", fontsize=7, color=C_GPU, style="italic")

    # ── GPU-resident memory graph (bottom inside GPU) ──
    mem_box = FancyBboxPatch((4.8, 1.0), 7.9, 2.4,
                              boxstyle="round,pad=0.08",
                              facecolor="#fff3cd", edgecolor="#856404",
                              linewidth=1.8)
    ax.add_patch(mem_box)
    ax.text(8.75, 3.05, "GPU-Resident Memory Graph (CSR Format)",
            ha="center", fontsize=10, fontweight="bold", color="#856404")

    # CSR components
    csr_items = [
        (5.0, 1.3, "row_offsets\n(N+1) × int32",   "#ffeaa7"),
        (6.7, 1.3, "col_indices\nE × int32",        "#ffeaa7"),
        (8.4, 1.3, "embeddings\nN×D × float32",     "#a8e6cf"),
        (10.1, 1.3, "modalities\nN × int32",        "#ffb3ba"),
        (11.5, 1.3, "timestamps\nN × float32",      "#bae1ff"),
    ]
    for x, y, label, color in csr_items:
        box = Rectangle((x, y), 1.3, 1.4, facecolor=color,
                        edgecolor="#34495e", linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.65, y + 0.7, label, ha="center", va="center",
                fontsize=7.5)

    # Arrows from kernels down to memory graph
    for x, _, _, color in stages:
        ax.annotate("", xy=(x + 0.9, 3.45), xytext=(x + 0.9, 4.35),
                    arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.4, alpha=0.7))

    # ── Output (right, outside GPU box) ──
    out_x = 13.5
    out_box = FancyBboxPatch((out_x, 4.5), 1.3, 1.5,
                              boxstyle="round,pad=0.06",
                              facecolor=C_ACCENT, edgecolor=C_ACCENT,
                              linewidth=1.8, alpha=0.9)
    ax.add_patch(out_box)
    ax.text(out_x + 0.65, 5.45, "Top-K\nResults",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    ax.text(out_x + 0.65, 4.82, "{id, score,\nmodality,\nhops}",
            ha="center", va="center", fontsize=6.5,
            style="italic", color="white")
    # Arrow from last kernel's right edge to results box
    k4_right = stages[-1][0] + 1.8  # 11.4 + 1.8 = 13.2
    ax.annotate("", xy=(out_x, 5.3), xytext=(k4_right + 0.05, 5.3),
                arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=2.5))

    plt.tight_layout()
    import os
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_pipeline.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_pipeline.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 2: MULTIMODAL GRAPH WITH CROSS-MODAL BRIDGES
# ════════════════════════════════════════════════════════════════
def make_graph():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0, 1.25, "Multimodal Neural Shortcut Network",
            fontsize=15, fontweight="bold", ha="center", color=C_GPU)
    ax.text(0, 1.15, "Ring lattice + skip connections + hubs + cross-modal bridges",
            fontsize=9, ha="center", color="#666", style="italic")

    # 30 nodes placed on a ring, split across modalities
    N = 30
    n_text, n_audio, n_image = 12, 9, 9
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, N, endpoint=False)
    R = 0.85
    pos = np.array([[R * np.cos(a), R * np.sin(a)] for a in angles])
    mods = ([0] * n_text) + ([1] * n_audio) + ([2] * n_image)
    np.random.seed(3)
    # Shuffle modality labels a bit so we see mixed modalities around the ring
    # but keep clusters
    colors_map = {0: C_TEXT, 1: C_AUDIO, 2: C_IMAGE}

    # ── Edges ──
    def draw_edge(i, j, color, lw=0.8, alpha=0.45, zorder=1, curve=0.0):
        if curve == 0:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=color, lw=lw, alpha=alpha, zorder=zorder)
        else:
            arr = FancyArrowPatch(pos[i], pos[j],
                                  connectionstyle=f"arc3,rad={curve}",
                                  arrowstyle="-", color=color,
                                  lw=lw, alpha=alpha, zorder=zorder)
            ax.add_patch(arr)

    # Ring (local neighbors k/2 = 3)
    for i in range(N):
        for j in (1, 2, 3):
            draw_edge(i, (i + j) % N, "#95a5a6", lw=0.7, alpha=0.5)

    # Skip connections (powers of 2)
    for lvl in range(2, 5):
        step = 1 << lvl
        for i in range(0, N, step):
            draw_edge(i, (i + step) % N, "#3498db",
                      lw=1.3, alpha=0.55, zorder=2, curve=0.22)

    # Hub nodes at sqrt(N) ≈ 5
    hubs = list(range(0, N, 5))
    for h in hubs:
        for offset in (7, 13, 19):
            t = (h + offset) % N
            draw_edge(h, t, C_HUB, lw=1.1, alpha=0.55, zorder=3, curve=0.4)

    # Cross-modal bridges — each modality group gets 2 bridges to each other
    text_ids  = [i for i in range(N) if mods[i] == 0]
    audio_ids = [i for i in range(N) if mods[i] == 1]
    image_ids = [i for i in range(N) if mods[i] == 2]
    bridges = [
        (text_ids[2], audio_ids[3]),
        (text_ids[5], audio_ids[7]),
        (text_ids[8], image_ids[2]),
        (text_ids[10], image_ids[6]),
        (audio_ids[1], image_ids[4]),
        (audio_ids[5], image_ids[8]),
    ]
    for a, b in bridges:
        draw_edge(a, b, C_ACCENT, lw=2.2, alpha=0.85, zorder=5, curve=0.55)

    # ── Nodes ──
    for i in range(N):
        is_hub = i in hubs
        color = colors_map[mods[i]]
        size = 320 if is_hub else 200
        edge_color = "#f1c40f" if is_hub else "white"
        edge_lw = 2.8 if is_hub else 1.5
        ax.scatter(pos[i, 0], pos[i, 1], s=size, c=color,
                   edgecolors=edge_color, linewidths=edge_lw, zorder=10)
        ax.text(pos[i, 0] * 1.12, pos[i, 1] * 1.12, str(i),
                fontsize=7, ha="center", va="center", color="#333", zorder=11)

    # ── Legend ──
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Text memory",
               markerfacecolor=C_TEXT, markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Audio memory",
               markerfacecolor=C_AUDIO, markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Image memory",
               markerfacecolor=C_IMAGE, markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Hub supernode",
               markerfacecolor="#7f8c8d", markeredgecolor="#f1c40f",
               markeredgewidth=2.5, markersize=13),
        Line2D([0], [0], color="#95a5a6", lw=1.2, label="Ring lattice"),
        Line2D([0], [0], color="#3498db", lw=1.5, label="Skip connection"),
        Line2D([0], [0], color=C_HUB, lw=1.3, label="Hub long-range"),
        Line2D([0], [0], color=C_ACCENT, lw=2.3, label="Cross-modal bridge"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8,
              frameon=True, facecolor="white", edgecolor="#bdc3c7")

    plt.tight_layout()
    import os
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_graph.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_graph.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 3: CUDA KERNEL DATAFLOW & WARP ORGANIZATION
# ════════════════════════════════════════════════════════════════
def make_kernels():
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.0, 9.15, "Optimized CUDA Kernel Pipeline — Measured on A100 PCIE",
            fontsize=15, fontweight="bold", ha="center", color=C_GPU)
    ax.text(7.0, 8.85, "Pre-allocated QueryContext  |  Explicit CUDA stream  |  GPU-side seed init",
            fontsize=8.5, ha="center", color="#666", style="italic")

    # ── Kernel 1: Cosine similarity ──
    ax.text(3.2, 8.35, "① Cosine Similarity  (0.017 ms @ N=4K)",
            fontsize=10.5, fontweight="bold", color="#2980b9")
    ax.add_patch(FancyBboxPatch((0.3, 5.4), 6.2, 2.75,
                  boxstyle="round,pad=0.08",
                  facecolor="#e8f4f8", edgecolor="#2980b9", linewidth=1.8))

    for i in range(4):
        for j in range(6):
            x = 0.6 + j * 0.5
            y = 7.3 - i * 0.35
            rect = Rectangle((x, y), 0.42, 0.27,
                             facecolor="#2980b9", alpha=0.85, edgecolor="white")
            ax.add_patch(rect)
    ax.text(0.6, 5.95, "Grid: N blocks", fontsize=8, fontweight="bold", color="#2980b9")
    ax.text(0.6, 5.7, "(one block per memory)", fontsize=7, color="#555")

    ax.text(4.1, 7.65, "Per block:", fontsize=8, color="#333")
    ax.text(4.1, 7.4, "256 threads = 8 warps × 32", fontsize=8, fontweight="bold", color="#333")
    ax.text(4.1, 7.0, "Each thread loads D/256 dims", fontsize=7.5, color="#555")
    ax.text(4.1, 6.75, "→ partial dot product", fontsize=7.5, color="#555")
    ax.text(4.1, 6.50, "→ __shfl_down_sync warp reduce", fontsize=7.5, color="#555")
    ax.text(4.1, 6.25, "→ block-level sum", fontsize=7.5, color="#555")
    ax.text(4.1, 5.92, "→ sim[node_id] = dot(q, emb)", fontsize=7.5, fontweight="bold", color="#2980b9")

    ax.add_patch(Rectangle((0.55, 5.52), 5.7, 0.22, facecolor="#2980b9", alpha=0.3))
    ax.text(3.4, 5.63, "__shfl_down_sync(0xFFFFFFFF, val, offset)",
            fontsize=7.5, ha="center", fontweight="bold", color="#2980b9")

    # ── Kernel 2: Temporal rerank ──
    ax.text(10.5, 8.35, "② Temporal Rerank  (0.008 ms)",
            fontsize=10.5, fontweight="bold", color="#16a085")
    ax.add_patch(FancyBboxPatch((7.0, 7.3), 6.5, 0.85,
                  boxstyle="round,pad=0.06",
                  facecolor="#e8f8f5", edgecolor="#16a085", linewidth=1.8))
    ax.text(10.25, 7.72, "score[i] = sim[i] × exp(−λ · age[i])   |   1 thread per memory   |   O(N)",
            fontsize=7.5, ha="center", fontweight="bold", color="#16a085")

    # ── Kernel 3: Tiled two-pass Top-K ──
    ax.text(10.5, 7.0, "③ Tiled Two-Pass Top-K  (0.41 ms @ N=4K)",
            fontsize=10.5, fontweight="bold", color="#8e44ad")
    ax.add_patch(FancyBboxPatch((7.0, 5.0), 6.5, 1.85,
                  boxstyle="round,pad=0.08",
                  facecolor="#f4ecf7", edgecolor="#8e44ad", linewidth=1.8))

    scores = np.random.default_rng(1).uniform(0.2, 1.0, size=16)
    scores[3] = 0.95; scores[7] = 0.92; scores[11] = 0.88
    for i, s in enumerate(scores):
        x = 7.25 + i * 0.36
        h = 0.35 * s + 0.1
        color = "#e74c3c" if s > 0.85 else "#9b59b6"
        ax.add_patch(Rectangle((x, 6.4), 0.32, h, facecolor=color, alpha=0.85))
    ax.text(10.25, 6.85, "scores[] (length N, partitioned into tiles of 16K)",
            fontsize=7, ha="center", color="#8e44ad", fontweight="bold")

    ax.add_patch(Rectangle((7.4, 5.85), 2.6, 0.42, facecolor="#8e44ad", alpha=0.2,
                            edgecolor="#8e44ad", lw=1))
    ax.text(8.7, 6.06, "Pass 1: per-tile K winners", fontsize=7, ha="center",
            fontweight="bold", color="#8e44ad")
    ax.annotate("", xy=(10.25, 6.06), xytext=(10.05, 6.06),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5))
    ax.add_patch(Rectangle((10.25, 5.85), 2.9, 0.42, facecolor="#e74c3c", alpha=0.2,
                            edgecolor="#c0392b", lw=1))
    ax.text(11.7, 6.06, "Pass 2: merge → final K", fontsize=7, ha="center",
            fontweight="bold", color="#c0392b")

    ax.add_patch(Rectangle((7.8, 5.15), 5.2, 0.55,
                  facecolor="#e74c3c", alpha=0.8, edgecolor="#c0392b", linewidth=1.5))
    ax.text(10.4, 5.42, "TOP-K SEEDS  (K=10)   →   bfs_init_seeds_kernel",
            fontsize=8.5, ha="center", va="center", color="white", fontweight="bold")

    # ── Kernel 4+5: BFS init + expansion ──
    ax.text(7.0, 4.7, "④ GPU-Side BFS Init  +  ⑤ Warp-Cooperative BFS Expansion  (0.085 ms)",
            fontsize=10.5, fontweight="bold", ha="center", color="#c0392b")

    ax.add_patch(FancyBboxPatch((0.3, 0.3), 13.2, 4.1,
                  boxstyle="round,pad=0.08",
                  facecolor="#fdecea", edgecolor="#c0392b", linewidth=1.8))

    # BFS init kernel (new!)
    ax.add_patch(FancyBboxPatch((0.6, 3.3), 4.2, 0.85,
                  boxstyle="round,pad=0.05",
                  facecolor="#f5b041", edgecolor="#e67e22", linewidth=1.5, alpha=0.9))
    ax.text(2.7, 3.85, "bfs_init_seeds_kernel (GPU-side)", fontsize=7.5,
            ha="center", fontweight="bold", color="#7d3c00")
    ax.text(2.7, 3.55, "reads top-K directly on device → populates",
            fontsize=6.5, ha="center", color="#7d3c00")
    ax.text(2.7, 3.35, "frontier[], hop_count[], bfs_score[]",
            fontsize=6.5, ha="center", fontweight="bold", color="#7d3c00")

    ax.annotate("", xy=(5.0, 3.7), xytext=(4.8, 3.7),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))

    # Frontier nodes
    ax.text(5.3, 4.0, "Frontier:", fontsize=7.5, color="#c0392b", fontweight="bold")
    for i in range(5):
        x = 5.3 + i * 0.55
        ax.add_patch(Circle((x + 0.2, 3.65), 0.2,
                             facecolor="#c0392b", edgecolor="white", lw=1.5))
        ax.text(x + 0.2, 3.65, str(i), fontsize=6, ha="center",
                va="center", color="white", fontweight="bold")

    warp_y = 2.55
    for i in range(5):
        x = 5.3 + i * 0.55 + 0.2
        ax.annotate("", xy=(x, warp_y + 0.35), xytext=(x, 3.42),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
        ax.add_patch(Rectangle((x - 0.2, warp_y - 0.15), 0.4, 0.5,
                               facecolor="#c0392b", alpha=0.25,
                               edgecolor="#c0392b", lw=1))
        ax.text(x, warp_y + 0.13, "warp", fontsize=5.5, ha="center",
                color="#c0392b", fontweight="bold")
        ax.text(x, warp_y - 0.05, "32 ln", fontsize=5, ha="center", color="#c0392b")

    ax.text(8.6, 2.7, "each warp's 32 lanes stride through",
            fontsize=7.5, color="#333", style="italic")
    ax.text(8.6, 2.45, "its node's CSR neighbor list",
            fontsize=7.5, color="#333", style="italic")
    ax.text(8.6, 2.15, "for (i = start + lane_id; i < end; i += 32)",
            fontsize=7, color="#333", family="monospace")

    ax.text(0.7, 1.75, "Race-free neighbor claiming:",
            fontsize=7.5, color="#333", fontweight="bold")
    ax.text(0.7, 1.45,
            "  if (atomicCAS(&hop_count[nb], -1, hop+1) == -1) {",
            fontsize=7, family="monospace", color="#2c3e50")
    ax.text(0.7, 1.2,
            "      bfs_score[nb] = max(parent * decay, sim * 0.8);",
            fontsize=7, family="monospace", color="#2c3e50")
    ax.text(0.7, 0.95,
            "      frontier_out[atomicAdd(count, 1)] = nb;",
            fontsize=7, family="monospace", color="#2c3e50")
    ax.text(0.7, 0.7, "  }",
            fontsize=7, family="monospace", color="#2c3e50")

    # Primitives box
    ax.add_patch(FancyBboxPatch((9.5, 0.5), 3.7, 1.55,
                  boxstyle="round,pad=0.06",
                  facecolor="white", edgecolor="#bdc3c7", linewidth=1, alpha=0.9))
    ax.text(11.35, 1.82, "CUDA Primitives Used:", fontsize=7.5,
            ha="center", fontweight="bold", color="#c0392b")
    primitives = [
        "• atomicCAS  (race-free claim)",
        "• atomicAdd  (frontier build)",
        "• __shfl_down_sync  (warp reduce)",
        "• __expf  (temporal decay)",
        "• cudaMemcpyAsync  (H2D query)",
    ]
    for i, prim in enumerate(primitives):
        ax.text(11.35, 1.58 - i * 0.2, prim, fontsize=6.5,
                ha="center", color="#333")

    # Pipeline flow arrows between sections
    ax.annotate("", xy=(7.0, 7.72), xytext=(6.5, 6.8),
                arrowprops=dict(arrowstyle="-|>", color=C_GPU, lw=2, alpha=0.6))
    ax.annotate("", xy=(10.25, 7.25), xytext=(10.25, 6.85),
                arrowprops=dict(arrowstyle="-|>", color=C_GPU, lw=2, alpha=0.6))
    ax.annotate("", xy=(10.4, 5.0), xytext=(10.4, 5.15),
                arrowprops=dict(arrowstyle="-|>", color=C_GPU, lw=2, alpha=0.6))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    import os
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_kernels.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_kernels.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 4: BENCHMARK RESULTS
# ════════════════════════════════════════════════════════════════
def make_benchmark():
    import os
    root = os.path.join(os.path.dirname(__file__), "..")

    measured_path = os.path.join(root, "results", "a100-pcie-40gb-v3", "results.json")
    if not os.path.exists(measured_path):
        measured_path = os.path.join(root, "results", "results.json")
    projected_path = os.path.join(root, "benchmark.json")
    if os.path.exists(measured_path):
        with open(measured_path) as f:
            raw = json.load(f)
        configs = raw["configurations"]
        gpu_name = raw["device"]["name"]
        sizes = [c["N"] for c in configs]
        sim  = [c["avg_latency_ms"]["similarity"] for c in configs]
        rer  = [c["avg_latency_ms"]["rerank"]     for c in configs]
        topk = [c["avg_latency_ms"]["topk"]       for c in configs]
        bfs  = [c["avg_latency_ms"]["bfs"]        for c in configs]
        totals = [c["avg_latency_ms"]["total"]    for c in configs]
        qps_vals = [c["throughput_qps"]           for c in configs]
        label_prefix = f"Measured on {gpu_name}"
        y_label = f"Measured Latency (ms, {gpu_name})"
    else:
        with open(projected_path) as f:
            results = json.load(f)
        sizes  = [r["N"] for r in results]
        sim    = [r["gpu_est"]["similarity"] for r in results]
        rer    = [r["gpu_est"]["rerank"]     for r in results]
        topk   = [r["gpu_est"]["topk"]       for r in results]
        bfs    = [r["gpu_est"]["bfs"]        for r in results]
        totals = [r["gpu_est"]["total"]      for r in results]
        qps_vals = [1000.0 / t for t in totals]
        label_prefix = "Projected (A100)"
        y_label = "Projected GPU Latency (ms, A100)"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x = np.arange(len(sizes))
    width = 0.6
    ax.bar(x, sim,  width, label="Cosine Similarity", color="#2980b9")
    ax.bar(x, rer,  width, bottom=sim, label="Temporal Rerank", color="#16a085")
    bot2 = [a + b for a, b in zip(sim, rer)]
    ax.bar(x, topk, width, bottom=bot2, label="Top-K Selection", color="#8e44ad")
    bot3 = [a + b for a, b in zip(bot2, topk)]
    ax.bar(x, bfs,  width, bottom=bot3, label="BFS Expansion", color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Memory Corpus Size (nodes)")
    ax.set_ylabel(y_label)
    ax.set_title("Per-Stage Pipeline Latency", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, t in enumerate(totals):
        ax.text(i, t + max(totals) * 0.02, f"{t:.2f}ms",
                ha="center", fontsize=8, fontweight="bold", color="#333")

    ax = axes[1]
    bars = ax.bar(x, qps_vals, width, color="#1a1a2e",
                  edgecolor="#f1c40f", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Memory Corpus Size (nodes)")
    ax.set_ylabel("Throughput (queries / second)")
    ax.set_title("Retrieval Throughput", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, q in enumerate(qps_vals):
        ax.text(i, q + max(qps_vals) * 0.015, f"{int(q):,}",
                ha="center", fontsize=9, fontweight="bold", color="#c0392b")

    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_benchmark.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_benchmark.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 5: SCALING SWEEP — LATENCY VS CORPUS SIZE
# ════════════════════════════════════════════════════════════════
def make_scaling():
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(root, "results", "a100-pcie-40gb-v3")
    if not os.path.isdir(results_dir):
        results_dir = os.path.join(root, "results")

    corpus_sizes = [1000, 5000, 10000, 20000, 50000]
    labels = ["1K", "5K", "10K", "20K", "50K"]
    files = [f"scale_{l.lower()}.json" for l in labels]

    wall_p50, wall_p99, gpu_p99, overhead = [], [], [], []
    for fname in files:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found, skipping scaling chart")
            return
        with open(path) as f:
            d = json.load(f)
        wall_p50.append(d["wall_latency_ms"]["p50"])
        wall_p99.append(d["wall_latency_ms"]["p99"])
        gpu_p99.append(d["gpu_kernel_ms"]["p99"])
        overhead.append(d["wall_latency_ms"]["p99"] - d["gpu_kernel_ms"]["p99"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(corpus_sizes))

    # Left: latency curves
    ax1.plot(x, wall_p99, "o-", color="#c0392b", lw=2.5, markersize=8,
             label="Wall-clock p99", zorder=5)
    ax1.plot(x, gpu_p99, "s--", color="#2980b9", lw=2, markersize=7,
             label="GPU kernel p99", zorder=5)
    ax1.plot(x, wall_p50, "^:", color="#7f8c8d", lw=1.5, markersize=6,
             label="Wall-clock p50", zorder=4)

    ax1.axhline(y=1.0, color="#e74c3c", linestyle="-", lw=1.5, alpha=0.7,
                label="1 ms budget (AV/Robot)")
    ax1.axhspan(0, 1.0, alpha=0.06, color="#2ecc71")
    ax1.axhspan(1.0, 3.0, alpha=0.04, color="#e74c3c")

    for i, (w, g) in enumerate(zip(wall_p99, gpu_p99)):
        ax1.annotate(f"{w:.2f}", (x[i], w), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7.5,
                     fontweight="bold", color="#c0392b")
        ax1.annotate(f"{g:.2f}", (x[i], g), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=7.5,
                     color="#2980b9")

    ax1.fill_between(x, gpu_p99, wall_p99, alpha=0.12, color="#e67e22",
                     label="Launch overhead")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s:,}" for s in corpus_sizes])
    ax1.set_xlabel("Corpus Size (memories)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency Scaling at 60 Hz (15s sustained)", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(wall_p99) * 1.15)

    # Right: overhead breakdown (stacked bars)
    ax2.bar(x, gpu_p99, 0.55, label="GPU kernel", color="#2980b9", alpha=0.85)
    ax2.bar(x, overhead, 0.55, bottom=gpu_p99, label="Launch overhead",
            color="#e67e22", alpha=0.85)

    ax2.axhline(y=1.0, color="#e74c3c", linestyle="-", lw=1.5, alpha=0.7)
    ax2.text(len(x) - 0.5, 1.05, "1 ms budget", fontsize=7.5, color="#e74c3c",
             ha="right")

    for i, (g, o) in enumerate(zip(gpu_p99, overhead)):
        total = g + o
        ax2.text(i, total + 0.04, f"{o:.2f}",
                 ha="center", fontsize=7.5, color="#e67e22", fontweight="bold")
        ax2.text(i, g / 2, f"{g:.2f}",
                 ha="center", fontsize=7.5, color="white", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:,}" for s in corpus_sizes])
    ax2.set_xlabel("Corpus Size (memories)")
    ax2.set_ylabel("p99 Latency (ms)")
    ax2.set_title("Overhead Breakdown (wall = GPU + launch)", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(wall_p99) * 1.15)

    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_scaling.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_scaling.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 6: SUSTAINED VS SHORT-RUN COMPARISON
# ════════════════════════════════════════════════════════════════
def make_sustained():
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(root, "results", "a100-pcie-40gb-v3")
    if not os.path.isdir(results_dir):
        results_dir = os.path.join(root, "results")

    workloads = [
        {"name": "AV\n60 Hz",    "short": "bench_av.json",    "long": "sustained_av_30s.json",
         "budget": 1.0, "short_n": "2.4K", "long_n": "5K"},
        {"name": "Robot\n1 kHz",  "short": "bench_robot.json",  "long": "sustained_robot_15s.json",
         "budget": 1.0, "short_n": "6K",   "long_n": "10K"},
        {"name": "AR/VR\n90 Hz",  "short": "bench_ar.json",    "long": "sustained_ar_30s.json",
         "budget": 5.0, "short_n": "20K",  "long_n": "50K"},
        {"name": "Voice\n30 Hz",  "short": "bench_voice.json",  "long": "sustained_voice_30s.json",
         "budget": 20.0, "short_n": "3K",  "long_n": "10K"},
    ]

    short_wall, long_wall, short_gpu, long_gpu, budgets = [], [], [], [], []
    names, short_ns, long_ns = [], [], []
    for w in workloads:
        sp = os.path.join(results_dir, w["short"])
        lp = os.path.join(results_dir, w["long"])
        if not os.path.exists(sp) or not os.path.exists(lp):
            print(f"  ⚠ Missing {w['short']} or {w['long']}, skipping sustained chart")
            return
        with open(sp) as f:
            sd = json.load(f)
        with open(lp) as f:
            ld = json.load(f)
        short_wall.append(sd["wall_latency_ms"]["p99"])
        long_wall.append(ld["wall_latency_ms"]["p99"])
        short_gpu.append(sd["gpu_kernel_ms"]["p99"])
        long_gpu.append(ld["gpu_kernel_ms"]["p99"])
        budgets.append(w["budget"])
        names.append(w["name"])
        short_ns.append(w["short_n"])
        long_ns.append(w["long_n"])

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(workloads))
    width = 0.18

    bars1 = ax.bar(x - 1.5 * width, short_gpu, width, label=f"Short-run GPU p99",
                   color="#2980b9", alpha=0.7, edgecolor="#2471a3", linewidth=1)
    bars2 = ax.bar(x - 0.5 * width, short_wall, width, label=f"Short-run wall p99",
                   color="#2980b9", alpha=0.95, edgecolor="#1a5276", linewidth=1)
    bars3 = ax.bar(x + 0.5 * width, long_gpu, width, label=f"Sustained GPU p99",
                   color="#e67e22", alpha=0.7, edgecolor="#ca6f1e", linewidth=1)
    bars4 = ax.bar(x + 1.5 * width, long_wall, width, label=f"Sustained wall p99",
                   color="#e67e22", alpha=0.95, edgecolor="#a04000", linewidth=1)

    for i, b in enumerate(budgets):
        bx = x[i]
        ax.plot([bx - 2.2 * width, bx + 2.2 * width], [b, b],
                color="#e74c3c", lw=2, zorder=10)
        if i == 0:
            ax.plot([], [], color="#e74c3c", lw=2, label="Deadline budget")

    def annotate_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold", rotation=45)
    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)
    annotate_bars(bars4)

    # Corpus size annotations
    for i in range(len(workloads)):
        ax.text(x[i] - width, -0.18, f"N={short_ns[i]}", ha="center",
                fontsize=7, color="#2980b9", fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(x[i] + width, -0.18, f"N={long_ns[i]}", ha="center",
                fontsize=7, color="#e67e22", fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Short-Run vs Sustained Benchmarks — A100 PCIE", fontweight="bold",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    max_val = max(max(short_wall), max(long_wall), max(budgets[:2]))
    ax.set_ylim(0, max_val * 1.35)

    fig.subplots_adjust(bottom=0.18)
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_sustained.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_sustained.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 7: WALL-CLOCK VS GPU KERNEL ACROSS ALL WORKLOADS
# ════════════════════════════════════════════════════════════════
def make_wallclock():
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(root, "results", "a100-pcie-40gb-v3")
    if not os.path.isdir(results_dir):
        results_dir = os.path.join(root, "results")

    benchmarks = [
        {"name": "AV\n2.4K / 60Hz",      "file": "bench_av.json",             "budget": 1.0},
        {"name": "AV 30s\n5K / 60Hz",     "file": "sustained_av_30s.json",     "budget": 1.0},
        {"name": "Robot\n6K / 1kHz",       "file": "bench_robot.json",          "budget": 1.0},
        {"name": "Robot 15s\n10K / 1kHz",  "file": "sustained_robot_15s.json",  "budget": 1.0},
        {"name": "AR\n20K / 90Hz",         "file": "bench_ar.json",             "budget": 5.0},
        {"name": "AR 30s\n50K / 90Hz",     "file": "sustained_ar_30s.json",     "budget": 5.0},
        {"name": "Voice\n3K / 30Hz",       "file": "bench_voice.json",          "budget": 20.0},
        {"name": "Voice 30s\n10K / 30Hz",  "file": "sustained_voice_30s.json",  "budget": 20.0},
    ]

    gpu_vals, overhead_vals, budgets_vals, names_list = [], [], [], []
    verdicts = []
    for b in benchmarks:
        path = os.path.join(results_dir, b["file"])
        if not os.path.exists(path):
            print(f"  ⚠ {b['file']} not found, skipping wallclock chart")
            return
        with open(path) as f:
            d = json.load(f)
        gpu = d["gpu_kernel_ms"]["p99"]
        wall = d["wall_latency_ms"]["p99"]
        gpu_vals.append(gpu)
        overhead_vals.append(wall - gpu)
        budgets_vals.append(b["budget"])
        names_list.append(b["name"])
        verdicts.append(wall <= b["budget"])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(benchmarks))
    width = 0.55

    ax.bar(x, gpu_vals, width, label="GPU kernel p99",
           color="#2980b9", alpha=0.9)
    ax.bar(x, overhead_vals, width, bottom=gpu_vals,
           label="Host overhead (launch + sync)",
           color="#e67e22", alpha=0.85)

    max_bar = max(g + o for g, o in zip(gpu_vals, overhead_vals))
    y_cap = max_bar * 1.35

    for i in range(len(benchmarks)):
        total = gpu_vals[i] + overhead_vals[i]
        color = "#2ecc71" if verdicts[i] else "#e74c3c"
        marker = "PASS" if verdicts[i] else "FAIL"
        ax.text(i, total + y_cap * 0.02, f"{total:.2f}\n{marker}",
                ha="center", fontsize=7, fontweight="bold", color=color)

    seen_budgets = set()
    for i, b in enumerate(budgets_vals):
        if b not in seen_budgets:
            start = i
            end = i
            while end + 1 < len(budgets_vals) and budgets_vals[end + 1] == b:
                end += 1
            if b <= y_cap:
                ax.plot([start - 0.4, end + 0.4], [b, b],
                        color="#e74c3c", lw=2, linestyle="--", zorder=10)
                ax.text(end + 0.4, b + y_cap * 0.01, f"{b:.0f} ms",
                        fontsize=7.5, color="#e74c3c", fontweight="bold", va="bottom")
            else:
                mid = (start + end) / 2
                ax.annotate(f"budget: {b:.0f} ms ↑",
                            xy=(mid, y_cap * 0.95), ha="center",
                            fontsize=7.5, color="#e74c3c", fontweight="bold")
            seen_budgets.add(b)

    ax.set_xticks(x)
    ax.set_xticklabels(names_list, fontsize=7.5)
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Wall-Clock Latency Breakdown — All Benchmarks on A100 PCIE",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_cap)

    fig.subplots_adjust(bottom=0.18)
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_wallclock.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_wallclock.png")


def make_opt_history():
    """Optimization history: rounds 1-3 + current for AV and Robot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    versions = ["Round 1\n(baseline)", "Round 2\n(targeted)", "Round 3\n(device)", "MARS\n(current)"]
    x = np.arange(len(versions))
    bar_w = 0.35

    # AV benchmark (N=2.4K, budget 1.0 ms)
    av_gpu  = [0.93, 0.74, 0.68, 0.10]
    av_wall = [1.41, 0.96, 0.87, 0.26]

    ax = axes[0]
    ax.bar(x - bar_w/2, av_gpu,  bar_w, label="GPU kernel p99", color="#3498db", alpha=0.85)
    ax.bar(x + bar_w/2, av_wall, bar_w, label="Wall-clock p99", color="#e67e22", alpha=0.85)
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="1 ms budget")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("AV Perception (60 Hz, N=2.4K)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, (g, w) in enumerate(zip(av_gpu, av_wall)):
        ax.text(i - bar_w/2, g + 0.03, f"{g:.2f}", ha="center", fontsize=8, fontweight="bold")
        color = "#27ae60" if w <= 1.0 else "#e74c3c"
        ax.text(i + bar_w/2, w + 0.03, f"{w:.2f}", ha="center", fontsize=8,
                fontweight="bold", color=color)

    ax.annotate("FAIL", xy=(0 + bar_w/2, av_wall[0]), xytext=(0.6, 1.55),
                fontsize=9, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
    ax.annotate("PASS", xy=(1 + bar_w/2, av_wall[1]), xytext=(1.6, 1.15),
                fontsize=9, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))

    # Robot benchmark (N=6K, budget 1.0 ms)
    robot_gpu  = [0.74, 1.09, 0.69, 0.17]
    robot_wall = [1.12, 1.21, 0.76, 0.34]

    ax = axes[1]
    ax.bar(x - bar_w/2, robot_gpu,  bar_w, label="GPU kernel p99", color="#3498db", alpha=0.85)
    ax.bar(x + bar_w/2, robot_wall, bar_w, label="Wall-clock p99", color="#e67e22", alpha=0.85)
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="1 ms budget")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Robot Episodic (1 kHz, N=6K)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, (g, w) in enumerate(zip(robot_gpu, robot_wall)):
        ax.text(i - bar_w/2, g + 0.03, f"{g:.2f}", ha="center", fontsize=8, fontweight="bold")
        color = "#27ae60" if w <= 1.0 else "#e74c3c"
        ax.text(i + bar_w/2, w + 0.03, f"{w:.2f}", ha="center", fontsize=8,
                fontweight="bold", color=color)

    ax.annotate("regression", xy=(1 - bar_w/2, robot_gpu[1]), xytext=(0.5, 1.50),
                fontsize=8, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
    ax.annotate("PASS", xy=(2 + bar_w/2, robot_wall[2]), xytext=(2.35, 0.45),
                fontsize=9, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))

    import os
    fig.suptitle("Optimization Progression: Four Rounds on A100",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_opt_history.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_opt_history.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 9: LARGE CORPUS SCALING (FP16+Graph)
# ════════════════════════════════════════════════════════════════
def make_large_scaling():
    """Large corpus scaling: 50K to 1M on RTX 5060 Ti and A100 SXM4."""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")

    corpus_sizes = [50000, 100000, 200000, 500000, 1000000]
    labels = ["50K", "100K", "200K", "500K", "1M"]

    # RTX 5060 Ti results
    rtx_dir = os.path.join(root, "results", "rtx5060ti-v4")
    a100_dir = os.path.join(root, "results", "a100-sxm4-80gb-v4")

    def load_large(results_dir, sizes):
        p99s = []
        fnames = ["large_50k.json", "large_100k.json", "large_200k.json",
                   "large_500k.json", "large_1m.json"]
        for fname in fnames:
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                p99s.append(None)
                continue
            with open(path) as f:
                d = json.load(f)
            p99s.append(d["wall_latency_ms"]["p99"])
        return p99s

    rtx_p99 = load_large(rtx_dir, corpus_sizes)
    a100_p99 = load_large(a100_dir, corpus_sizes)

    if all(v is None for v in rtx_p99) and all(v is None for v in a100_p99):
        print("  ⚠ No large corpus results found, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(corpus_sizes))
    bar_w = 0.35

    # Left: line chart
    if any(v is not None for v in rtx_p99):
        vals = [v if v is not None else 0 for v in rtx_p99]
        ax1.plot(x, vals, "o-", color="#2ecc71", lw=2.5, markersize=8,
                 label="RTX 5060 Ti ($449)", zorder=5)
        for i, v in enumerate(vals):
            if v > 0:
                ax1.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8,
                             fontweight="bold", color="#2ecc71")

    if any(v is not None for v in a100_p99):
        vals = [v if v is not None else 0 for v in a100_p99]
        ax1.plot(x, vals, "s-", color="#3498db", lw=2.5, markersize=8,
                 label="A100 SXM4 ($15K)", zorder=5)
        for i, v in enumerate(vals):
            if v > 0:
                ax1.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                             xytext=(0, -15), ha="center", fontsize=8,
                             fontweight="bold", color="#3498db")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Corpus Size (memories)")
    ax1.set_ylabel("Wall-clock p99 Latency (ms)")
    ax1.set_title("Large Corpus Scaling (FP16 + CUDA Graph, 60 Hz)",
                  fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # Right: bar chart comparison at 1M
    gpus = ["RTX 5060 Ti\n($449)", "A100 SXM4\n($15K)"]
    vals_1m = [rtx_p99[-1] or 0, a100_p99[-1] or 0]
    colors = ["#2ecc71", "#3498db"]
    bars = ax2.bar(gpus, vals_1m, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals_1m):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{v:.1f} ms", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Wall-clock p99 Latency (ms)")
    ax2.set_title("1M Memories — GPU Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_large_scaling.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_large_scaling.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 10: FP16 vs FP32 A/B COMPARISON
# ════════════════════════════════════════════════════════════════
def make_fp16_comparison():
    """FP32 vs FP16 vs FP16+Graph at N=10K and N=50K across GPUs."""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")

    gpu_dirs = {
        "A100 PCIE":   os.path.join(root, "results", "a100-pcie-40gb-v4"),
        "A100 SXM4":   os.path.join(root, "results", "a100-sxm4-80gb-v4"),
        "RTX 5060 Ti": os.path.join(root, "results", "rtx5060ti-v4"),
    }

    configs = [
        ("FP32",       "fp32_50k.json"),
        ("FP16",       "fp16_50k.json"),
        ("FP16+Graph", "fp16_graph_50k.json"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    gpu_colors = {"A100 PCIE": "#e74c3c", "A100 SXM4": "#3498db", "RTX 5060 Ti": "#2ecc71"}
    bar_w = 0.25
    x = np.arange(len(configs))

    for gi, (gpu_name, gpu_dir) in enumerate(gpu_dirs.items()):
        vals = []
        for cfg_name, fname in configs:
            path = os.path.join(gpu_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                vals.append(d["wall_latency_ms"]["p99"])
            else:
                vals.append(0)

        if all(v == 0 for v in vals):
            continue

        offset = (gi - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=gpu_name,
                      color=gpu_colors[gpu_name], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs])
    ax.set_ylabel("Wall-clock p99 Latency (ms)")
    ax.set_title("FP32 vs FP16 vs FP16+Graph — N=50K, 60 Hz", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_fp16_comparison.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ diag_fp16_comparison.png")


# ════════════════════════════════════════════════════════════════
#  DIAGRAM 11: FAISS / CAGRA COMPARISON
# ════════════════════════════════════════════════════════════════
def make_faiss_comparison():
    """Grouped bar chart: FAISS Flat, FAISS IVF, cuVS CAGRA, MARS across corpus sizes."""
    import os

    corpus_labels = ["N=2.4K", "N=10K", "N=20K", "N=50K"]

    # Wall-clock p99 ms on A100 SXM4 80GB, D=768
    faiss_flat = [0.10, 0.12, 0.18, 0.35]
    faiss_ivf  = [0.13, 0.15, 0.30, 0.28]
    cagra      = [2.60, 2.29, 2.29, 2.47]
    mars_latency    = [0.26, 0.34, 0.36, 0.44]

    systems = [
        ("FAISS Flat",  faiss_flat, "#e74c3c"),
        ("FAISS IVF",   faiss_ivf,  "#e67e22"),
        ("cuVS CAGRA",  cagra,      "#9b59b6"),
        ("MARS",     mars_latency,    "#2ecc71"),
    ]

    n_groups = len(corpus_labels)
    n_bars = len(systems)
    bar_w = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    for si, (name, vals, color) in enumerate(systems):
        offset = (si - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=name, color=color, alpha=0.88)
        for bar, v in enumerate(vals):
            bx = x[bar] + offset
            ax.text(bx, vals[bar] + 0.04, f"{vals[bar]:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(corpus_labels, fontsize=10)
    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Single-query p99 Latency (ms)")
    ax.set_title("Same-Hardware Comparison \u2014 A100 SXM4 80GB, D=768",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Note about feature difference
    ax.text(0.98, 0.97, "Ours adds cross-modal + temporal decay",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, style="italic", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#ccc", alpha=0.9))

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(outdir, "diag_faiss_comparison.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  \u2713 diag_faiss_comparison.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    make_pipeline()
    make_graph()
    make_kernels()
    make_benchmark()
    make_scaling()
    make_sustained()
    make_wallclock()
    make_opt_history()
    make_large_scaling()
    make_fp16_comparison()
    make_faiss_comparison()
    print("All 11 diagrams generated.")
