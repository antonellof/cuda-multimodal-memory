#!/usr/bin/env python3
"""
Parse ablation study results from results/ablation/*.json into a
publication-ready table for the paper.

Usage:
    python3 scripts/parse_ablation.py [results_dir]

Output: Markdown and LaTeX tables showing recall@10, cross-modal hit rate,
and wall-clock p99 latency for each ablation variant and corpus size.
"""
import json
import os
import sys
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"  Skipping {path.name} (parse error)", file=sys.stderr)
            continue

        cfg = data.get("config", {})
        wall = data.get("wall_latency_ms", {})
        recall = data.get("recall", {})

        rows.append({
            "variant": cfg.get("ablation", "unknown"),
            "N": cfg.get("corpus_size", 0),
            "bfs_hops": cfg.get("bfs_max_hops", 2),
            "wall_p99": wall.get("p99", 0),
            "wall_p50": wall.get("p50", 0),
            "gpu_p99": data.get("gpu_kernel_ms", {}).get("p99", 0),
            "recall_at_10": recall.get("recall_at_10", -1),
            "cross_modal_rate": recall.get("cross_modal_rate", -1),
            "file": path.name,
        })
    return rows


def print_markdown_table(rows: list[dict]):
    # Group by N
    corpus_sizes = sorted(set(r["N"] for r in rows))
    variants = ["full", "no_bridges", "no_hubs", "h0", "h1", "h3", "flat"]

    print("\n## Ablation Results\n")
    for N in corpus_sizes:
        print(f"\n### N = {N:,}\n")
        print("| Variant | BFS h | Wall p99 (ms) | GPU p99 (ms) | Recall@10 | Cross-modal |")
        print("|---------|-------|---------------|-------------|-----------|-------------|")
        for v in variants:
            matches = [r for r in rows if r["variant"] == v and r["N"] == N]
            if not matches:
                continue
            r = matches[0]
            recall_str = f"{r['recall_at_10']:.2f}" if r["recall_at_10"] >= 0 else "—"
            cm_str = f"{r['cross_modal_rate']:.2f}" if r["cross_modal_rate"] >= 0 else "—"
            print(f"| {v:13s} | {r['bfs_hops']:5d} | {r['wall_p99']:13.4f} | {r['gpu_p99']:11.4f} | {recall_str:9s} | {cm_str:11s} |")


def print_latex_table(rows: list[dict]):
    """Print a LaTeX table suitable for the paper."""
    corpus_sizes = sorted(set(r["N"] for r in rows))
    variants = [
        ("full", "Full NSN ($h{=}2$)"),
        ("no_bridges", "No bridges"),
        ("no_hubs", "No hubs"),
        ("h0", "Full NSN, $h{=}0$"),
        ("h1", "Full NSN, $h{=}1$"),
        ("h3", "Full NSN, $h{=}3$"),
        ("flat", "Flat brute-force"),
    ]

    print("\n% ── LaTeX ablation table (paste into paper) ──")
    print("\\begin{table}[ht]")
    print("\\centering\\small")
    print("\\caption{NSN ablation study. Recall@10 is measured against a CPU")
    print("brute-force reference; cross-modal rate is the fraction of queries")
    print("returning results from $\\geq 2$ modalities.}")
    print("\\label{tab:ablation}")

    # Build column spec
    n_cols = len(corpus_sizes)
    col_spec = "l" + "rrr" * n_cols
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Header row 1: corpus sizes
    header = "\\textbf{Variant}"
    for N in corpus_sizes:
        header += f" & \\multicolumn{{3}}{{c}}{{$N{{{=}}}{N // 1000}$K}}"
    header += " \\\\"
    print(header)

    # Header row 2: metrics
    subheader = ""
    for _ in corpus_sizes:
        subheader += " & \\textbf{$p99$} & \\textbf{R@10} & \\textbf{CM}"
    subheader += " \\\\"
    print("\\cmidrule(l){2-" + str(1 + 3 * n_cols) + "}")
    print(subheader)
    print("\\midrule")

    for v_key, v_label in variants:
        line = v_label
        for N in corpus_sizes:
            matches = [r for r in rows if r["variant"] == v_key and r["N"] == N]
            if matches:
                r = matches[0]
                p99 = f"{r['wall_p99']:.2f}"
                recall = f"{r['recall_at_10']:.2f}" if r["recall_at_10"] >= 0 else "---"
                cm = f"{r['cross_modal_rate']:.2f}" if r["cross_modal_rate"] >= 0 else "---"
                line += f" & {p99} & {recall} & {cm}"
            else:
                line += " & --- & --- & ---"
        line += " \\\\"
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/ablation"
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} not found. Run `make bench-ablation` first.",
              file=sys.stderr)
        sys.exit(1)

    rows = load_results(results_dir)
    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} results from {results_dir}")
    print_markdown_table(rows)
    print()
    print_latex_table(rows)


if __name__ == "__main__":
    main()
