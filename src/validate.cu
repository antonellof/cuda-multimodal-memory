// ============================================================================
//  validate.cu
//  Hardware validation harness for MARS.
//
//  Purpose: when run on a real NVIDIA GPU (e.g. rented via vast.ai), this
//  produces a JSON report containing:
//
//    1. GPU device properties (name, compute capability, SM count, HBM, ...)
//    2. Per-stage kernel timings measured with cudaEvents across all
//       corpus sizes from the projection table (N = 1K..16K)
//    3. Correctness sanity checks (ground-truth top-K vs BFS-expanded top-K)
//    4. Memory footprint measured via cudaMemGetInfo
//    5. Environment metadata (driver version, CUDA runtime version)
//
//  Output is written to stdout as a single JSON document so that CI pipelines
//  or shell scripts can parse it directly:
//
//      ./validate > results.json
//
//  Build: part of the main Makefile — `make validate`
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <string>

#include "memory_graph.h"
#include "memory_cuda.cuh"

namespace {

// ─── JSON writer (minimal, hand-rolled to avoid dependencies) ────────
struct Json {
    std::string out;
    int indent = 0;

    void pad() { for (int i = 0; i < indent; ++i) out += "  "; }
    void open(const char* key, char bracket) {
        pad();
        if (key) { out += "\""; out += key; out += "\": "; }
        out += bracket; out += "\n";
        ++indent;
    }
    void close(char bracket, bool last = false) {
        --indent;
        pad();
        out += bracket;
        if (!last) out += ",";
        out += "\n";
    }
    void kv_str(const char* key, const std::string& val, bool last = false) {
        pad();
        out += "\""; out += key; out += "\": \"";
        for (char c : val) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        out += "\"";
        if (!last) out += ",";
        out += "\n";
    }
    void kv_int(const char* key, long long v, bool last = false) {
        pad(); out += "\""; out += key; out += "\": " + std::to_string(v);
        if (!last) out += ","; out += "\n";
    }
    void kv_float(const char* key, double v, int digits, bool last = false) {
        char buf[64]; snprintf(buf, sizeof(buf), "%.*f", digits, v);
        pad(); out += "\""; out += key; out += "\": "; out += buf;
        if (!last) out += ","; out += "\n";
    }
    void kv_bool(const char* key, bool v, bool last = false) {
        pad(); out += "\""; out += key; out += "\": "; out += v ? "true" : "false";
        if (!last) out += ","; out += "\n";
    }
};

// ─── Emit GPU device properties ─────────────────────────────────────
void emit_device_info(Json& j) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    j.open("device", '{');

    if (device_count == 0) {
        j.kv_str("error", "No CUDA devices found", true);
        j.close('}');
        return;
    }

    int current_device = 0;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    j.kv_str  ("name",                 prop.name);
    j.kv_int  ("compute_capability_major", prop.major);
    j.kv_int  ("compute_capability_minor", prop.minor);
    j.kv_int  ("multi_processor_count",    prop.multiProcessorCount);
    j.kv_int  ("warp_size",                prop.warpSize);
    j.kv_int  ("max_threads_per_block",    prop.maxThreadsPerBlock);
    j.kv_int  ("total_global_mem_bytes",   (long long)prop.totalGlobalMem);
    j.kv_float("total_global_mem_gb",      prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), 2);
    j.kv_int  ("memory_clock_khz",         prop.memoryClockRate);
    j.kv_int  ("memory_bus_width_bits",    prop.memoryBusWidth);
    // Theoretical peak bandwidth = 2 * memClock(kHz) * busWidth(bits) / 8 / 1e6 GB/s
    double peak_bw = 2.0 * prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8.0 / 1e9;
    j.kv_float("peak_bandwidth_gb_per_s",  peak_bw, 1);
    j.kv_int  ("driver_version",           driver_version);
    j.kv_int  ("runtime_version",          runtime_version, true);
    j.close('}');
}

// ─── Ground-truth CPU top-K for correctness check ───────────────────
std::vector<int32_t> cpu_top_k(const std::vector<float>& embeddings,
                               const std::vector<float>& query,
                               int32_t n, int32_t d, int32_t k) {
    std::vector<std::pair<float, int32_t>> scored(n);
    for (int32_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (int32_t j = 0; j < d; ++j)
            dot += double(embeddings[size_t(i) * d + j]) * query[j];
        scored[i] = { float(dot), i };
    }
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int32_t> ids(k);
    for (int32_t i = 0; i < k; ++i) ids[i] = scored[i].second;
    return ids;
}

// ─── Run one corpus-size configuration and emit a JSON object ───────
void benchmark_config(Json& j, int32_t n_text, int32_t n_audio, int32_t n_image,
                      int32_t dim, int32_t n_queries, bool last_item) {
    int32_t N = n_text + n_audio + n_image;

    // Build graph
    auto t0 = std::chrono::high_resolution_clock::now();
    auto graph = MemoryGraph::synthetic(n_text, n_audio, n_image, dim);
    graph.build_nsn_edges(6, 0.15);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Upload
    size_t free_before, total_mem;
    cudaMemGetInfo(&free_before, &total_mem);
    t0 = std::chrono::high_resolution_clock::now();
    DeviceMemoryGraph dg = upload_to_device(graph);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t free_after;
    cudaMemGetInfo(&free_after, &total_mem);
    double vram_used_mb = (free_before - free_after) / (1024.0 * 1024.0);

    // Warm up
    std::mt19937 rng(12345);
    std::normal_distribution<float> N01(0.0f, 1.0f);
    auto make_query = [&]() {
        std::vector<float> q(dim);
        float norm = 0.0f;
        for (int i = 0; i < dim; ++i) { q[i] = N01(rng); norm += q[i]*q[i]; }
        norm = std::sqrt(norm) + 1e-8f;
        for (int i = 0; i < dim; ++i) q[i] /= norm;
        return q;
    };
    QueryContext ctx = create_query_context(N, dim, 64);
    {
        auto q = make_query();
        RetrievalConfig cfg;
        RetrievalStats s;
        auto results = query_memory_fast(dg, ctx, q.data(), 1e7f, cfg, s);
        (void)results;
    }
    cudaDeviceSynchronize();

    // Accumulate timings over n_queries runs
    double acc_sim = 0, acc_rer = 0, acc_topk = 0, acc_bfs = 0, acc_total = 0;
    int32_t acc_waves = 0;
    int32_t n_results_total = 0;

    // Correctness: for query 0, compare top-10 with CPU ground truth
    auto q0 = make_query();
    int32_t correctness_hits = 0;
    int32_t correctness_k = 10;
    {
        auto gt = cpu_top_k(graph.embeddings, q0, N, dim, correctness_k);
        RetrievalConfig cfg;
        cfg.top_k = correctness_k;
        RetrievalStats s;
        auto res = query_memory_fast(dg, ctx, q0.data(), 1e7f, cfg, s);
        for (auto& r : res) {
            for (auto id : gt) if (r.node_id == id) { ++correctness_hits; break; }
        }
    }

    for (int32_t q = 0; q < n_queries; ++q) {
        auto qv = make_query();
        RetrievalConfig cfg;
        RetrievalStats s;
        auto res = query_memory_fast(dg, ctx, qv.data(), 1e7f, cfg, s);
        acc_sim   += s.gpu_ms_similarity;
        acc_rer   += s.gpu_ms_rerank;
        acc_topk  += s.gpu_ms_topk;
        acc_bfs   += s.gpu_ms_bfs;
        acc_total += s.gpu_ms_total;
        acc_waves += s.bfs_waves;
        n_results_total += (int32_t)res.size();
    }

    double total_avg = acc_total / n_queries;
    double qps = total_avg > 0 ? 1000.0 / total_avg : 0.0;

    j.open(nullptr, '{');
    j.kv_int  ("N",                   N);
    j.kv_int  ("n_text",              n_text);
    j.kv_int  ("n_audio",             n_audio);
    j.kv_int  ("n_image",             n_image);
    j.kv_int  ("embedding_dim",       dim);
    j.kv_int  ("num_edges",           graph.num_edges / 2);
    j.kv_float("avg_degree",          double(graph.num_edges) / N, 2);
    j.kv_float("build_ms",            build_ms, 3);
    j.kv_float("upload_ms",           upload_ms, 3);
    j.kv_float("vram_used_mb",        vram_used_mb, 2);
    j.kv_int  ("n_queries",           n_queries);
    j.open("avg_latency_ms", '{');
    j.kv_float("similarity", acc_sim   / n_queries, 4);
    j.kv_float("rerank",     acc_rer   / n_queries, 4);
    j.kv_float("topk",       acc_topk  / n_queries, 4);
    j.kv_float("bfs",        acc_bfs   / n_queries, 4);
    j.kv_float("total",      total_avg, 4, true);
    j.close('}');
    j.kv_float("throughput_qps",      qps, 1);
    j.kv_float("avg_bfs_waves",       double(acc_waves) / n_queries, 2);
    j.kv_float("avg_results_per_query", double(n_results_total) / n_queries, 1);
    j.open("correctness", '{');
    j.kv_int  ("top_k_tested",     correctness_k);
    j.kv_int  ("seeds_matched",    correctness_hits);
    j.kv_float("recall_at_k",      double(correctness_hits) / correctness_k, 3, true);
    j.close('}', true);
    j.close('}', last_item);

    destroy_query_context(ctx);
    free_device(dg);
}

} // namespace

int main(int argc, char** argv) {
    // Parse optional flags
    int32_t n_queries = 100;
    int32_t dim       = 768;
    if (argc > 1) n_queries = std::atoi(argv[1]);
    if (argc > 2) dim       = std::atoi(argv[2]);

    Json j;
    j.open(nullptr, '{');
    j.kv_str("tool",           "cuda-multimodal-memory-validate");
    j.kv_str("version",        "0.1.0");
    j.kv_int("n_queries_per_config", n_queries);
    j.kv_int("embedding_dim",  dim);

    emit_device_info(j);

    j.open("configurations", '[');

    struct Cfg { int32_t nt, na, ni; };
    std::vector<Cfg> configs = {
        {   500,   250,   250 },
        {  1000,   500,   500 },
        {  2000,  1000,  1000 },
        {  4000,  2000,  2000 },
        {  8000,  4000,  4000 },
    };
    for (size_t i = 0; i < configs.size(); ++i) {
        benchmark_config(j, configs[i].nt, configs[i].na, configs[i].ni,
                         dim, n_queries, i == configs.size() - 1);
    }

    j.close(']', true);
    j.close('}', true);

    std::fputs(j.out.c_str(), stdout);
    return 0;
}
