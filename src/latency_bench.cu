// ============================================================================
//  latency_bench.cu
//
//  Standalone deadline-aware benchmark tool for MARS. Unlike validate.cu
//  (which reports aggregate latencies over
//  back-to-back queries), this tool simulates a fixed-rate sensor feed by
//  issuing queries at exact frame boundaries and reports the metrics that
//  matter for real-time workloads:
//
//    - p50, p90, p99, p99.9, p99.99, max
//    - inter-query jitter (stddev of latency)
//    - deadline miss rate at a budget
//    - frame over-budget count
//
//  Exit code is non-zero if the observed p99 exceeds the configured budget,
//  so the tool integrates directly into CI pipelines.
//
//  Usage:
//    ./latency_bench [rate_hz] [budget_ms] [frames] [corpus_size]
//    ./latency_bench  60        1.0         600      2400       # AV perception
//    ./latency_bench 1000       1.0         10000    6000       # Humanoid 1kHz
//    ./latency_bench   90       5.0         4500     20000      # AR/VR 90Hz
//    ./latency_bench   30      20.0         900      3000       # Voice agent
//
//  Output is a single JSON document on stdout so that CI can parse it.
// ============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "memory_graph.h"
#include "memory_cuda.cuh"
#include "cmng.cuh"

namespace {

// ─── Latency histogram ──────────────────────────────────────────────
struct Histogram {
    std::vector<double> samples;

    void add(double ms) { samples.push_back(ms); }

    double percentile(double p) const {
        if (samples.empty()) return 0.0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
        return sorted[idx];
    }

    double p50()    const { return percentile(0.50); }
    double p90()    const { return percentile(0.90); }
    double p99()    const { return percentile(0.99); }
    double p999()   const { return percentile(0.999); }
    double p9999()  const { return percentile(0.9999); }
    double max_ms() const {
        double m = 0.0;
        for (double s : samples) if (s > m) m = s;
        return m;
    }

    double mean() const {
        if (samples.empty()) return 0.0;
        double s = 0.0;
        for (double v : samples) s += v;
        return s / samples.size();
    }

    double stddev() const {
        if (samples.size() < 2) return 0.0;
        double m = mean();
        double acc = 0.0;
        for (double v : samples) acc += (v - m) * (v - m);
        return std::sqrt(acc / (samples.size() - 1));
    }

    int32_t count_above(double budget_ms) const {
        int32_t c = 0;
        for (double v : samples) if (v > budget_ms) ++c;
        return c;
    }
};

// ─── Ablation variant names ──────────────────────────────────────────
// "full"        — all 5 NSN phases (baseline)
// "no_bridges"  — phases 1-4, no cross-modal bridges
// "no_hubs"     — phases 1,2,4,5, no hub supernodes
// "no_rewire"   — phases 1,2,3,5, no small-world rewiring
// "ring_only"   — phase 1 only (ring lattice)
// "flat"        — no graph edges at all (brute-force top-K only)

struct AblationConfig {
    std::string name        = "full";
    bool enable_ring        = true;
    bool enable_skips       = true;
    bool enable_hubs        = true;
    bool enable_rewire      = true;
    bool enable_bridges     = true;
    int32_t bfs_max_hops    = 2;
};

AblationConfig parse_ablation(const char* name) {
    AblationConfig a;
    a.name = name;
    if (std::strcmp(name, "full") == 0) {
        // default — all phases, h=2
    } else if (std::strcmp(name, "no_bridges") == 0) {
        a.enable_bridges = false;
    } else if (std::strcmp(name, "no_hubs") == 0) {
        a.enable_hubs = false;
    } else if (std::strcmp(name, "no_rewire") == 0) {
        a.enable_rewire = false;
    } else if (std::strcmp(name, "ring_only") == 0) {
        a.enable_skips = false; a.enable_hubs = false;
        a.enable_rewire = false; a.enable_bridges = false;
    } else if (std::strcmp(name, "flat") == 0) {
        a.enable_ring = false; a.enable_skips = false;
        a.enable_hubs = false; a.enable_rewire = false;
        a.enable_bridges = false; a.bfs_max_hops = 0;
    } else if (std::strcmp(name, "h0") == 0) {
        a.bfs_max_hops = 0;
    } else if (std::strcmp(name, "h1") == 0) {
        a.bfs_max_hops = 1;
    } else if (std::strcmp(name, "h3") == 0) {
        a.bfs_max_hops = 3;
    } else {
        std::fprintf(stderr, "Unknown ablation: %s\n", name);
        std::exit(1);
    }
    return a;
}

// ─── CPU brute-force top-K for recall measurement ───────────────────
std::vector<int32_t> cpu_topk(const float* embeddings, int32_t N,
                               int32_t D, const float* query, int32_t K) {
    std::vector<std::pair<float, int32_t>> scores(N);
    for (int32_t i = 0; i < N; ++i) {
        float dot = 0.0f;
        for (int32_t d = 0; d < D; ++d)
            dot += embeddings[size_t(i) * D + d] * query[d];
        scores[i] = {dot, i};
    }
    std::partial_sort(scores.begin(), scores.begin() + K, scores.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int32_t> result(K);
    for (int32_t i = 0; i < K; ++i) result[i] = scores[i].second;
    return result;
}

// ─── Cross-modal hit rate: fraction of results with ≥2 modalities ──
double cross_modal_rate(const std::vector<QueryResult>& results,
                         const std::vector<int32_t>& modalities,
                         int32_t check_top = 10) {
    std::unordered_set<int32_t> mods;
    int32_t n = std::min(int32_t(results.size()), check_top);
    for (int32_t i = 0; i < n; ++i) {
        if (results[i].node_id >= 0 &&
            results[i].node_id < int32_t(modalities.size()))
            mods.insert(modalities[results[i].node_id]);
    }
    return mods.size() >= 2 ? 1.0 : 0.0;
}

// ─── Fill a corpus with L2-normalized random embeddings ─────────────
void populate_corpus(MemoryGraph& g, int32_t n_total, int32_t dim,
                     const AblationConfig& ablation, uint32_t seed = 42) {
    // Split as 50% text, 25% audio, 25% image for realistic mix
    int32_t n_text  = n_total / 2;
    int32_t n_audio = n_total / 4;
    int32_t n_image = n_total - n_text - n_audio;
    g = MemoryGraph::synthetic(n_text, n_audio, n_image, dim, seed);

    MemoryGraph::NSNConfig nsn;
    nsn.k              = 6;
    nsn.p              = 0.15;
    nsn.enable_ring    = ablation.enable_ring;
    nsn.enable_skips   = ablation.enable_skips;
    nsn.enable_hubs    = ablation.enable_hubs;
    nsn.enable_rewire  = ablation.enable_rewire;
    nsn.enable_bridges = ablation.enable_bridges;
    g.build_nsn_edges_configurable(nsn);
}

// ─── Produce a deterministic query vector for frame i ───────────────
void make_query(std::vector<float>& out, int32_t dim, int32_t frame_id) {
    std::mt19937 rng(0x5EED0000u + frame_id);
    std::normal_distribution<float> N01(0.0f, 1.0f);
    float norm = 0.0f;
    out.resize(dim);
    for (int32_t i = 0; i < dim; ++i) { out[i] = N01(rng); norm += out[i] * out[i]; }
    norm = std::sqrt(norm) + 1e-8f;
    for (int32_t i = 0; i < dim; ++i) out[i] /= norm;
}

// ─── Sleep until a specific wall-clock timepoint ───────────────────
using Clock = std::chrono::steady_clock;
using Timepoint = Clock::time_point;

void sleep_until(Timepoint t) {
    auto now = Clock::now();
    if (now >= t) return;
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t - now);
    if (diff.count() > 500) {
        std::this_thread::sleep_for(diff - std::chrono::microseconds(200));
    }
    while (Clock::now() < t) { }
}

// ─── GPU keepalive: tiny kernel to prevent clock-state drops ────────
__global__ void keepalive_kernel() {
    // 1 thread, 1 block — just enough to keep the GPU from idling.
    // Takes ~2µs to launch, prevents the ~0.1-0.2ms clock ramp penalty.
}

void keepalive_loop(cudaStream_t stream, Timepoint deadline) {
    constexpr int64_t KEEPALIVE_INTERVAL_US = 2000;  // every 2ms
    while (true) {
        auto now = Clock::now();
        if (now >= deadline - std::chrono::microseconds(500)) break;
        keepalive_kernel<<<1, 1, 0, stream>>>();
        auto next = now + std::chrono::microseconds(KEEPALIVE_INTERVAL_US);
        if (next >= deadline - std::chrono::microseconds(500)) break;
        std::this_thread::sleep_until(next);
    }
}

} // namespace

int main(int argc, char** argv) {
    // ─── Parse args ──────────────────────────────────────────────
    double  rate_hz     = 60.0;
    double  budget_ms   = 1.0;
    int32_t frames      = 600;
    int32_t corpus_size = 2400;
    int32_t dim         = 768;
    int32_t top_k       = 10;
    bool    keepalive   = false;
    bool    use_fp16    = false;
    bool    use_graph   = false;
    bool    use_cmng    = false;
    bool    use_mars      = false;
    AblationConfig ablation;
    bool    measure_recall = false;

    // Separate positional args from flags (--prefixed)
    std::vector<const char*> pos_args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--keepalive") == 0) { keepalive = true; continue; }
        if (std::strcmp(argv[i], "--fp16")      == 0) { use_fp16  = true; continue; }
        if (std::strcmp(argv[i], "--graph")     == 0) { use_graph = true; continue; }
        if (std::strcmp(argv[i], "--cmng")      == 0) { use_cmng  = true; continue; }
        if (std::strcmp(argv[i], "--v6") == 0 || std::strcmp(argv[i], "--mars")        == 0) { use_mars    = true; continue; }
        if (std::strcmp(argv[i], "--recall")    == 0) { measure_recall = true; continue; }
        if (std::strcmp(argv[i], "--ablate") == 0 && i + 1 < argc) {
            ablation = parse_ablation(argv[++i]); continue;
        }
        pos_args.push_back(argv[i]);
    }
    if (pos_args.size() > 0) rate_hz     = std::atof(pos_args[0]);
    if (pos_args.size() > 1) budget_ms   = std::atof(pos_args[1]);
    if (pos_args.size() > 2) frames      = std::atoi(pos_args[2]);
    if (pos_args.size() > 3) corpus_size = std::atoi(pos_args[3]);
    if (pos_args.size() > 4) dim         = std::atoi(pos_args[4]);

    const double frame_period_us = 1e6 / rate_hz;

    // ─── Print workload header to stderr (so stdout stays pure JSON) ─
    std::fprintf(stderr,
        "\n══════════════════════════════════════════════════════════\n"
        "  Real-time latency benchmark\n"
        "    rate     : %.1f Hz\n"
        "    budget   : %.3f ms (p99 must be under this)\n"
        "    frames   : %d (%.1f s of simulated sensor feed)\n"
        "    corpus   : %d memories  dim=%d  top_k=%d\n"
        "    keepalive: %s\n"
        "    fp16:      %s\n"
        "    graph:     %s\n"
        "    cmng:      %s\n"
        "    mars:      %s\n"
        "    ablation:  %s\n"
        "    bfs_hops:  %d\n"
        "    recall:    %s\n"
        "══════════════════════════════════════════════════════════\n\n",
        rate_hz, budget_ms, frames,
        frames / rate_hz, corpus_size, dim, top_k,
        keepalive ? "ON (GPU clock warm)" : "OFF",
        use_fp16  ? "ON (FP16 similarity)" : "OFF",
        use_graph ? "ON (CUDA graph replay)" : "OFF",
        use_cmng  ? "ON (CMNG beam search)" : "OFF",
        use_mars    ? "ON (cuBLAS sim + CUB top-K)" : "OFF",
        ablation.name.c_str(),
        ablation.bfs_max_hops,
        measure_recall ? "ON" : "OFF");

    // ─── Build + upload corpus ───────────────────────────────────
    MemoryGraph graph;
    populate_corpus(graph, corpus_size, dim, ablation);
    // ─── Build graph (CMNG or brute-force CSR) ─────────────────
    DeviceMemoryGraph dg = {};
    CMNGGraph cmng_g = {};
    CMNGSearchContext cmng_ctx = {};
    QueryContext ctx = {};

    if (use_cmng) {
        cmng_g = build_cmng(graph.embeddings.data(), graph.modalities.data(),
                            graph.timestamps.data(), corpus_size, dim,
                            /*degree=*/64, /*degree_cross=*/4);
        if (use_fp16) {
            cmng_upload_fp16(cmng_g);
            std::fprintf(stderr, "  CMNG FP16 embeddings uploaded (%.1f MB saved)\n",
                         float(corpus_size) * dim * 2.0f / 1e6f);
        }
        cmng_ctx = create_cmng_context(dim);
    } else {
        dg = upload_to_device(graph);
        if (use_fp16) {
            upload_fp16_embeddings(dg);
            std::fprintf(stderr, "  FP16 embeddings uploaded (%.1f MB saved)\n",
                         float(corpus_size) * dim * 2.0f / 1e6f);
        }
        ctx = create_query_context(corpus_size, dim, top_k);
        ctx.use_fp16       = use_fp16;
        ctx.use_cuda_graph = use_graph;
        ctx.use_cublas     = use_mars;
        ctx.use_cub_topk   = use_mars;
    }

    // ─── Warm up ─────────────────────────────────────────────────
    {
        std::vector<float> q;
        make_query(q, dim, -1);
        if (use_cmng) {
            CMNGSearchConfig cfg;
            cfg.K = top_k; cfg.use_fp16 = use_fp16;
            RetrievalStats stats;
            auto res = cmng_search(cmng_g, cmng_ctx, q.data(), 1e7f, cfg, stats);
            (void)res;
        } else {
            RetrievalConfig cfg; cfg.top_k = top_k;
            RetrievalStats  stats;
            auto res = query_memory_fast(dg, ctx, q.data(), 1e7f, cfg, stats);
            (void)res;
        }
    }
    cudaDeviceSynchronize();

    // ─── Run fixed-rate loop ─────────────────────────────────────
    Histogram total_hist;
    Histogram kernel_hist;
    int32_t deadline_misses = 0;
    double recall_sum = 0.0;
    double crossmodal_sum = 0.0;
    int32_t recall_count = 0;

    auto run_start = Clock::now();
    for (int32_t f = 0; f < frames; ++f) {
        auto frame_deadline = run_start +
            std::chrono::microseconds(int64_t((f + 1) * frame_period_us));

        std::vector<float> q;
        make_query(q, dim, f);
        RetrievalStats stats;

        auto t0 = Clock::now();
        std::vector<QueryResult> results;
        if (use_cmng) {
            CMNGSearchConfig cfg;
            cfg.K = top_k; cfg.use_fp16 = use_fp16;
            results = cmng_search(cmng_g, cmng_ctx, q.data(), float(f), cfg, stats);
        } else {
            RetrievalConfig cfg;
            cfg.top_k = top_k;
            cfg.bfs_max_hops = ablation.bfs_max_hops;
            results = query_memory_fast(dg, ctx, q.data(), float(f), cfg, stats);
        }
        auto t1 = Clock::now();

        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_hist.add(wall_ms);
        kernel_hist.add(stats.gpu_ms_total);

        if (wall_ms > budget_ms) ++deadline_misses;

        // Measure recall and cross-modal hit rate (sample every 10th frame)
        if (measure_recall && (f % 10 == 0)) {
            auto gt = cpu_topk(graph.embeddings.data(), corpus_size,
                               dim, q.data(), top_k);
            std::unordered_set<int32_t> gt_set(gt.begin(), gt.end());
            int32_t hits = 0;
            for (auto& r : results)
                if (gt_set.count(r.node_id)) ++hits;
            recall_sum += double(hits) / top_k;
            crossmodal_sum += cross_modal_rate(results, graph.modalities, top_k);
            ++recall_count;
        }

        // Yield the rest of the frame, with optional keepalive
        if (keepalive) {
            keepalive_loop(ctx.stream, frame_deadline);
        }
        sleep_until(frame_deadline);
    }

    if (use_cmng) {
        destroy_cmng_context(cmng_ctx);
        free_cmng(cmng_g);
    } else {
        destroy_query_context(ctx);
        free_device(dg);
    }

    // ─── Emit JSON report on stdout ──────────────────────────────
    double miss_rate = double(deadline_misses) / frames;
    bool   p99_ok    = total_hist.p99() <= budget_ms;

    std::printf("{\n");
    std::printf("  \"tool\": \"cuda-multimodal-memory-latency-bench\",\n");
    std::printf("  \"version\": \"0.1.0\",\n");
    std::printf("  \"config\": {\n");
    std::printf("    \"rate_hz\": %.1f,\n", rate_hz);
    std::printf("    \"budget_ms\": %.4f,\n", budget_ms);
    std::printf("    \"frames\": %d,\n", frames);
    std::printf("    \"corpus_size\": %d,\n", corpus_size);
    std::printf("    \"embedding_dim\": %d,\n", dim);
    std::printf("    \"top_k\": %d,\n", top_k);
    std::printf("    \"keepalive\": %s,\n", keepalive ? "true" : "false");
    std::printf("    \"fp16\": %s,\n", use_fp16 ? "true" : "false");
    std::printf("    \"cuda_graph\": %s,\n", use_graph ? "true" : "false");
    std::printf("    \"cmng\": %s,\n", use_cmng ? "true" : "false");
    std::printf("    \"mars\": %s,\n", use_mars ? "true" : "false");
    std::printf("    \"ablation\": \"%s\",\n", ablation.name.c_str());
    std::printf("    \"bfs_max_hops\": %d,\n", ablation.bfs_max_hops);
    std::printf("    \"measure_recall\": %s\n", measure_recall ? "true" : "false");
    std::printf("  },\n");
    std::printf("  \"wall_latency_ms\": {\n");
    std::printf("    \"p50\":    %.4f,\n", total_hist.p50());
    std::printf("    \"p90\":    %.4f,\n", total_hist.p90());
    std::printf("    \"p99\":    %.4f,\n", total_hist.p99());
    std::printf("    \"p99_9\":  %.4f,\n", total_hist.p999());
    std::printf("    \"p99_99\": %.4f,\n", total_hist.p9999());
    std::printf("    \"max\":    %.4f,\n", total_hist.max_ms());
    std::printf("    \"mean\":   %.4f,\n", total_hist.mean());
    std::printf("    \"stddev\": %.4f\n",  total_hist.stddev());
    std::printf("  },\n");
    std::printf("  \"gpu_kernel_ms\": {\n");
    std::printf("    \"p50\": %.4f,\n", kernel_hist.p50());
    std::printf("    \"p99\": %.4f,\n", kernel_hist.p99());
    std::printf("    \"max\": %.4f\n",  kernel_hist.max_ms());
    std::printf("  },\n");
    std::printf("  \"deadline\": {\n");
    std::printf("    \"budget_ms\":       %.4f,\n", budget_ms);
    std::printf("    \"deadline_misses\": %d,\n",   deadline_misses);
    std::printf("    \"miss_rate\":       %.6f,\n", miss_rate);
    std::printf("    \"p99_within_budget\": %s\n",  p99_ok ? "true" : "false");
    std::printf("  }");
    if (measure_recall && recall_count > 0) {
        double avg_recall = recall_sum / recall_count;
        double avg_crossmodal = crossmodal_sum / recall_count;
        std::printf(",\n  \"recall\": {\n");
        std::printf("    \"recall_at_%d\": %.4f,\n", top_k, avg_recall);
        std::printf("    \"cross_modal_rate\": %.4f,\n", avg_crossmodal);
        std::printf("    \"samples\": %d\n", recall_count);
        std::printf("  }");
    }
    std::printf("\n}\n");

    // ─── Summary to stderr ───────────────────────────────────────
    std::fprintf(stderr,
        "\n═══════════════════════════════════════════════════════════\n"
        "  Results\n"
        "    p50   : %8.4f ms\n"
        "    p99   : %8.4f ms %s\n"
        "    p99.9 : %8.4f ms\n"
        "    max   : %8.4f ms\n"
        "    jitter: %8.4f ms (stddev)\n"
        "    misses: %d / %d (%.3f%%)\n"
        "    %s\n"
        "═══════════════════════════════════════════════════════════\n\n",
        total_hist.p50(),
        total_hist.p99(),
        p99_ok ? "✓" : "✗ OVER BUDGET",
        total_hist.p999(),
        total_hist.max_ms(),
        total_hist.stddev(),
        deadline_misses, frames, 100.0 * miss_rate,
        p99_ok ? "PASS — p99 within deadline budget"
               : "FAIL — p99 exceeded deadline budget");

    return p99_ok ? 0 : 1;
}
