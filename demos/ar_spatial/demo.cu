// ============================================================================
//  demos/ar_spatial/demo.cu
//
//  Workload: AR/VR spatial memory (Tier 2 / perception-critical).
//
//  Scenario: an AR headset runs its perception pipeline at 90 Hz. Over the
//  course of a 5-minute session, SLAM landmarks are inserted into the
//  substrate with visual descriptors + 3D positions (~27 000 landmarks at
//  steady state). On every frame, the headset queries for landmarks in the
//  user's current gaze cone to pre-fetch cached 3D mesh data and anchor
//  content placement.
//
//  The frame budget is 11.1 ms total (for 90 Hz display refresh). Memory
//  retrieval gets at most 5 ms of that budget to leave room for the rest
//  of perception + rendering.
//
//  Success criteria:
//    - p99 retrieval latency < 5.0 ms over 4 500 frames (50 seconds)
//    - landmark "cache hit rate" > 90% (top-10 contains a relevant landmark)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "memory_graph.h"
#include "memory_cuda.cuh"
#include "frame_timer.h"

using namespace demo;

int main(int argc, char** argv) {
    // ── Configuration ──
    constexpr int32_t FRAME_RATE_HZ   = 90;
    constexpr int32_t NUM_FRAMES      = 4500;   // 50 seconds
    constexpr int32_t NUM_LANDMARK_CLUSTERS = 300;  // spatial regions
    constexpr int32_t CORPUS_SIZE     = 27000;  // 5 minutes of landmarks
    constexpr int32_t EMBEDDING_DIM   = 768;
    constexpr double  DEADLINE_P99_MS = 5.0;

    DemoReport report;
    report.demo_name       = "ar_spatial";
    report.scenario        = "90 Hz AR headset, 5-minute session, gaze-directed landmark pre-fetch";
    report.frame_rate_hz   = FRAME_RATE_HZ;
    report.deadline_p99_ms = DEADLINE_P99_MS;
    report.corpus_size     = CORPUS_SIZE;
    report.embedding_dim   = EMBEDDING_DIM;

    // ── Build landmark corpus ──
    // Landmarks cluster spatially: nearby landmarks have similar visual
    // descriptors (same wall, same room, same object). We simulate this by
    // sampling from NUM_LANDMARK_CLUSTERS canonical spatial regions.
    SyntheticSensor sensor(EMBEDDING_DIM, NUM_LANDMARK_CLUSTERS);

    MemoryGraph graph;
    graph.num_nodes     = CORPUS_SIZE;
    graph.embedding_dim = EMBEDDING_DIM;
    graph.embeddings.resize(size_t(CORPUS_SIZE) * EMBEDDING_DIM);
    graph.modalities.resize(CORPUS_SIZE, MOD_IMAGE);
    graph.timestamps.resize(CORPUS_SIZE);

    std::mt19937 rng(271);
    std::uniform_int_distribution<int32_t> cluster_dist(0, NUM_LANDMARK_CLUSTERS - 1);

    for (int32_t i = 0; i < CORPUS_SIZE; ++i) {
        int32_t cluster = cluster_dist(rng);
        auto emb = sensor.sample(cluster, 0.12f);
        std::memcpy(&graph.embeddings[size_t(i) * EMBEDDING_DIM],
                    emb.data(), EMBEDDING_DIM * sizeof(float));
        graph.timestamps[i] = float(i) / FRAME_RATE_HZ;  // in seconds
    }

    graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
    std::fprintf(stderr, "Graph built: %d nodes, %d edges\n",
                 graph.num_nodes, graph.num_edges / 2);

    DeviceMemoryGraph dg = upload_to_device(graph);
    QueryContext ctx = create_query_context(CORPUS_SIZE, EMBEDDING_DIM, 20);

    // ── Retrieval config tuned for the AR regime ──
    // Slightly larger top-K because the application wants multiple
    // candidate landmarks for prefetch.
    RetrievalConfig cfg;
    cfg.top_k             = 20;
    cfg.bfs_max_hops      = 2;
    cfg.time_decay_lambda = 1e-7f;  // temporal decay matters less for spatial
    cfg.bfs_score_decay   = 0.5f;
    cfg.modality_filter   = -1;

    // ── Frame loop: simulate gaze drifting through the scene ──
    // The user's gaze is modeled as a sequence of target clusters that the
    // gaze drifts toward and holds on for ~30 frames before switching.
    int32_t hits = 0;
    int32_t total = 0;
    int32_t current_gaze_target = cluster_dist(rng);
    int32_t frames_on_current_target = 0;

    for (int32_t frame = 0; frame < NUM_FRAMES; ++frame) {
        if (frames_on_current_target > 30) {
            current_gaze_target = cluster_dist(rng);
            frames_on_current_target = 0;
        }
        ++frames_on_current_target;

        auto query_emb = sensor.sample(current_gaze_target, 0.15f);
        float query_ts = float(CORPUS_SIZE + frame) / FRAME_RATE_HZ;

        RetrievalStats stats;
        auto results = query_memory_fast(dg, ctx, query_emb.data(), query_ts, cfg, stats);

        report.latencies.add(double(stats.gpu_ms_total));
        ++total;

        // Cache hit: any result in top-10 with score > 0.65
        bool hit = false;
        for (size_t i = 0; i < results.size() && i < 10; ++i) {
            if (results[i].score > 0.65f) { hit = true; break; }
        }
        if (hit) ++hits;
    }

    report.num_frames = NUM_FRAMES;
    report.add_metric("cache_hit_rate", double(hits) / total);
    report.add_metric("total_queries", double(total));

    report.print_summary();
    report.print_json();

    destroy_query_context(ctx);
    free_device(dg);
    return report.passed() ? 0 : 1;
}
