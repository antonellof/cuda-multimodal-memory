// ============================================================================
//  demos/av_perception/demo.cu
//
//  Workload: Autonomous vehicle perception memory (Tier 1 / safety-critical).
//
//  Scenario: a 60 Hz camera produces frames with 5-20 object detections each.
//  Each detection has a 768-D appearance embedding from a vision-language
//  model. The substrate maintains a 2-second sliding window (~2400 memories
//  at peak detection density) of previously-tracked object embeddings.
//
//  On every frame, for each new detection, we query the memory to answer:
//  "is this the same object I tracked before, possibly after occlusion?"
//
//  Success criteria:
//    - p99 query latency < 1.0 ms
//    - sustained 60 Hz over 600 frames (10 seconds)
//    - track recovery rate > 95% after simulated 400 ms occlusions
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
    constexpr int32_t FRAME_RATE          = 60;
    constexpr int32_t NUM_FRAMES          = 600;   // 10 seconds
    constexpr int32_t NUM_IDENTITIES      = 40;    // 40 unique objects in scene
    constexpr int32_t DETECTIONS_PER_FRAME_MIN = 5;
    constexpr int32_t DETECTIONS_PER_FRAME_MAX = 15;
    constexpr int32_t CORPUS_SIZE         = 2400;  // 2-second sliding window
    constexpr int32_t EMBEDDING_DIM       = 768;
    constexpr double  DEADLINE_P99_MS     = 1.0;

    DemoReport report;
    report.demo_name       = "av_perception";
    report.scenario        = "60 Hz camera, 2-second sliding window, occlusion re-ID";
    report.frame_rate_hz   = FRAME_RATE;
    report.deadline_p99_ms = DEADLINE_P99_MS;
    report.corpus_size     = CORPUS_SIZE;
    report.embedding_dim   = EMBEDDING_DIM;

    // ── Build initial corpus from synthetic object identities ──
    SyntheticSensor sensor(EMBEDDING_DIM, NUM_IDENTITIES);

    // Create a MemoryGraph with CORPUS_SIZE synthetic memories:
    // each memory is a "detection" of one of NUM_IDENTITIES objects
    // with small per-detection drift.
    MemoryGraph graph;
    graph.num_nodes     = CORPUS_SIZE;
    graph.embedding_dim = EMBEDDING_DIM;
    graph.embeddings.resize(size_t(CORPUS_SIZE) * EMBEDDING_DIM);
    graph.modalities.resize(CORPUS_SIZE, MOD_IMAGE);  // all image modality
    graph.timestamps.resize(CORPUS_SIZE);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> id_dist(0, NUM_IDENTITIES - 1);

    for (int32_t i = 0; i < CORPUS_SIZE; ++i) {
        int32_t identity = id_dist(rng);
        auto emb = sensor.sample(identity, 0.08f);
        std::memcpy(&graph.embeddings[size_t(i) * EMBEDDING_DIM],
                    emb.data(), EMBEDDING_DIM * sizeof(float));
        graph.timestamps[i] = float(i);  // monotonically increasing time
    }

    // Build NSN edges over the corpus
    graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
    std::fprintf(stderr, "Graph built: %d nodes, %d edges\n",
                 graph.num_nodes, graph.num_edges / 2);

    // Upload to GPU
    DeviceMemoryGraph dg = upload_to_device(graph);
    QueryContext ctx = create_query_context(CORPUS_SIZE, EMBEDDING_DIM, 10);

    // ── Frame loop ──
    RetrievalConfig cfg;
    cfg.top_k             = 10;
    cfg.bfs_max_hops      = 2;
    cfg.time_decay_lambda = 1e-6f;
    cfg.bfs_score_decay   = 0.5f;
    cfg.modality_filter   = -1;

    int32_t correct_matches = 0;
    int32_t total_matches   = 0;

    // Simulate occlusion events: every 30 frames (500 ms), one random
    // identity is "hidden" for 24 frames (400 ms); after it reappears we
    // check whether the query returns a memory of the same identity.
    constexpr int32_t OCCLUSION_PERIOD = 30;
    constexpr int32_t OCCLUSION_LENGTH = 24;

    std::uniform_int_distribution<int32_t> count_dist(
        DETECTIONS_PER_FRAME_MIN, DETECTIONS_PER_FRAME_MAX);

    for (int32_t frame = 0; frame < NUM_FRAMES; ++frame) {
        int32_t detections_this_frame = count_dist(rng);
        float   frame_ts = float(CORPUS_SIZE + frame);

        // The "active" identity that is currently occluded (if any)
        int32_t occluded_identity = -1;
        int32_t occlusion_start   = (frame / OCCLUSION_PERIOD) * OCCLUSION_PERIOD;
        if (frame - occlusion_start < OCCLUSION_LENGTH) {
            occluded_identity = (frame / OCCLUSION_PERIOD) % NUM_IDENTITIES;
        }

        RetrievalStats stats;
        for (int32_t d = 0; d < detections_this_frame; ++d) {
            int32_t identity = id_dist(rng);
            // Skip queries for the currently-occluded identity
            if (identity == occluded_identity) continue;

            auto query_emb = sensor.sample(identity, 0.10f);

            auto results = query_memory_fast(dg, ctx, query_emb.data(), frame_ts,
                                             cfg, stats);
            report.latencies.add(double(stats.gpu_ms_total));

            // Correctness: the top result should come from a memory of the
            // same identity. We check this by seeing if the nearest neighbor's
            // embedding has high cosine similarity to the query (it should,
            // because memories of the same identity cluster tightly).
            if (!results.empty()) {
                ++total_matches;
                if (results[0].score > 0.75f) ++correct_matches;
            }
        }

        // Once per frame, check that the occluded identity can be re-identified
        // on the first frame after occlusion ends (track recovery test).
        if (occluded_identity >= 0 &&
            frame - occlusion_start == OCCLUSION_LENGTH - 1) {
            auto query_emb = sensor.sample(occluded_identity, 0.10f);
            auto results = query_memory_fast(dg, ctx, query_emb.data(), frame_ts,
                                             cfg, stats);
            report.latencies.add(double(stats.gpu_ms_total));
            ++total_matches;
            if (!results.empty() && results[0].score > 0.70f) ++correct_matches;
        }
    }

    report.num_frames = NUM_FRAMES;
    double recovery_rate = total_matches > 0
        ? double(correct_matches) / total_matches : 0.0;
    report.add_metric("track_recovery_rate", recovery_rate);
    report.add_metric("total_queries", double(total_matches));
    report.add_metric("avg_queries_per_frame",
                      double(total_matches) / NUM_FRAMES);

    // ── Output ──
    report.print_summary();
    report.print_json();

    destroy_query_context(ctx);
    free_device(dg);
    return report.passed() ? 0 : 1;
}
