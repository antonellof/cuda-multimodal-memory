// ============================================================================
//  main.cu
//  Driver for the MARS benchmark.
//  Builds a synthetic corpus of text/audio/image memories, constructs an
//  NSN topology with cross-modal bridges, uploads to GPU, and runs a
//  batch of queries measuring per-stage latency.
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <string>

#include "memory_graph.h"
#include "memory_cuda.cuh"

namespace {
const char* mod_name(int32_t m) {
    switch (m) {
        case MOD_TEXT:  return "TEXT";
        case MOD_AUDIO: return "AUDIO";
        case MOD_IMAGE: return "IMAGE";
        default:        return "?";
    }
}
} // namespace

int main(int argc, char** argv) {
    // ── Configuration ────────────────────────────────────────────
    int32_t n_text  = 4000;
    int32_t n_audio = 2000;
    int32_t n_image = 2000;
    int32_t dim     = 768;      // typical CLIP / BERT embedding size
    int32_t queries = 100;

    if (argc > 1) n_text  = std::atoi(argv[1]);
    if (argc > 2) n_audio = std::atoi(argv[2]);
    if (argc > 3) n_image = std::atoi(argv[3]);
    if (argc > 4) dim     = std::atoi(argv[4]);

    std::printf("════════════════════════════════════════════════════════════\n");
    std::printf("  MARS — Benchmark\n");
    std::printf("════════════════════════════════════════════════════════════\n");

    // ── Build synthetic corpus ───────────────────────────────────
    std::printf("\n[1/4] Building synthetic multimodal corpus...\n");
    auto graph = MemoryGraph::synthetic(n_text, n_audio, n_image, dim);

    // ── Build NSN topology with multimodal bridges ───────────────
    std::printf("[2/4] Building Neural Shortcut Network edges...\n");
    graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
    graph.print_summary();

    // ── Upload to GPU ────────────────────────────────────────────
    std::printf("\n[3/4] Uploading to GPU...\n");
    DeviceMemoryGraph dg = upload_to_device(graph);
    std::printf("  ✓ %d nodes, %d edges, %d-D embeddings\n",
                dg.num_nodes, dg.num_edges / 2, dg.embedding_dim);

    // ── Generate queries and run retrieval ───────────────────────
    std::printf("\n[4/4] Running %d queries...\n\n", queries);
    std::mt19937 rng(123);
    std::normal_distribution<float> N01(0.0f, 1.0f);

    int32_t N = n_text + n_audio + n_image;
    QueryContext ctx = create_query_context(N, dim, 10);

    RetrievalConfig cfg;
    cfg.top_k           = 10;
    cfg.bfs_max_hops    = 2;
    cfg.time_decay_lambda = 1e-8f;
    cfg.bfs_score_decay = 0.5f;
    cfg.modality_filter = -1;

    float total_sim = 0, total_topk = 0, total_bfs = 0, total_rerank = 0, total_all = 0;

    for (int q = 0; q < queries; ++q) {
        std::vector<float> query(dim);
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) { query[d] = N01(rng); norm += query[d] * query[d]; }
        norm = std::sqrt(norm) + 1e-8f;
        for (int d = 0; d < dim; ++d) query[d] /= norm;

        float query_ts = 1e7f;
        RetrievalStats stats;
        auto results = query_memory_fast(dg, ctx, query.data(), query_ts, cfg, stats);

        total_sim    += stats.gpu_ms_similarity;
        total_topk   += stats.gpu_ms_topk;
        total_bfs    += stats.gpu_ms_bfs;
        total_rerank += stats.gpu_ms_rerank;
        total_all    += stats.gpu_ms_total;

        // Print first query in detail
        if (q == 0) {
            std::printf("  Sample query results (top %d of %d total retrieved):\n",
                        static_cast<int>(results.size()),
                        static_cast<int>(results.size()));
            for (size_t i = 0; i < results.size() && i < 10; ++i) {
                std::printf("    [%2zu] node=%5d  mod=%-5s  hop=%d  score=%.4f\n",
                            i + 1, results[i].node_id, mod_name(results[i].modality),
                            results[i].hops_from_seed, results[i].score);
            }
            std::printf("\n  Per-stage timing (first query):\n");
            std::printf("    Cosine similarity : %7.3f ms\n", stats.gpu_ms_similarity);
            std::printf("    Temporal rerank   : %7.3f ms\n", stats.gpu_ms_rerank);
            std::printf("    Top-K selection   : %7.3f ms\n", stats.gpu_ms_topk);
            std::printf("    BFS expansion     : %7.3f ms  (%d waves)\n",
                        stats.gpu_ms_bfs, stats.bfs_waves);
            std::printf("    TOTAL             : %7.3f ms\n\n", stats.gpu_ms_total);
        }
    }

    std::printf("──────────────────────────────────────────────────────────\n");
    std::printf("  Averaged over %d queries:\n", queries);
    std::printf("    Cosine similarity : %7.3f ms\n", total_sim    / queries);
    std::printf("    Temporal rerank   : %7.3f ms\n", total_rerank / queries);
    std::printf("    Top-K selection   : %7.3f ms\n", total_topk   / queries);
    std::printf("    BFS expansion     : %7.3f ms\n", total_bfs    / queries);
    std::printf("    TOTAL             : %7.3f ms\n", total_all    / queries);
    std::printf("    Throughput        : %7.1f queries/second\n",
                1000.0 / (total_all / queries));
    std::printf("──────────────────────────────────────────────────────────\n");

    destroy_query_context(ctx);
    free_device(dg);
    return 0;
}
