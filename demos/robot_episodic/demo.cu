// ============================================================================
//  demos/robot_episodic/demo.cu
//
//  Workload: Humanoid robot episodic memory (Tier 1 / safety-critical).
//  The hardest latency budget in the evaluation: 1 kHz, sub-1 ms p99.
//
//  Scenario: a whole-body controller runs at 1000 Hz. On every tick the
//  robot's joint state + local visual context is encoded into a 768-D
//  situation embedding. The substrate maintains a 10-second sliding
//  window of prior situations (up to 10 000 memories) tagged with
//  success/failure outcomes of whatever action was taken from that state.
//
//  On every tick, the controller queries: "what happened the last time
//  I was in a state similar to the current one?". The retrieval has to
//  complete in under 1 ms or the controller misses its deadline and a
//  safety supervisor interrupts the motion.
//
//  Success criteria:
//    - p99 query latency < 1.0 ms
//    - sustained 1 kHz over 10 000 queries (10 seconds of operation)
//    - retrieval returns a semantically related memory (score > 0.7)
//      on > 90% of queries
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
    constexpr int32_t CONTROL_RATE_HZ     = 1000;
    constexpr int32_t NUM_TICKS           = 10000;  // 10 seconds
    constexpr int32_t NUM_SITUATIONS      = 150;    // ~150 distinct states
    constexpr int32_t CORPUS_SIZE         = 10000;  // 10-second window
    constexpr int32_t EMBEDDING_DIM       = 768;
    constexpr double  DEADLINE_P99_MS     = 1.0;

    DemoReport report;
    report.demo_name       = "robot_episodic";
    report.scenario        = "1 kHz whole-body control loop, 10-second episodic window";
    report.frame_rate_hz   = CONTROL_RATE_HZ;
    report.deadline_p99_ms = DEADLINE_P99_MS;
    report.corpus_size     = CORPUS_SIZE;
    report.embedding_dim   = EMBEDDING_DIM;

    // ── Build a MemoryGraph of past situations ──
    // The robot has NUM_SITUATIONS canonical state types (standing, reaching,
    // grasping, walking, climbing stairs, etc.). Each stored memory is a
    // noisy sample from one of these canonical states.
    SyntheticSensor sensor(EMBEDDING_DIM, NUM_SITUATIONS);

    MemoryGraph graph;
    graph.num_nodes     = CORPUS_SIZE;
    graph.embedding_dim = EMBEDDING_DIM;
    graph.embeddings.resize(size_t(CORPUS_SIZE) * EMBEDDING_DIM);

    // Mixed modalities: joint-state embeddings are "audio-like" (CLAP-style
    // time-series), visual-context embeddings are "image", action outcomes
    // are "text". This exercises the cross-modal bridges.
    graph.modalities.resize(CORPUS_SIZE);
    graph.timestamps.resize(CORPUS_SIZE);

    std::mt19937 rng(137);
    std::uniform_int_distribution<int32_t> sit_dist(0, NUM_SITUATIONS - 1);
    std::uniform_int_distribution<int32_t> mod_dist(0, 2);

    for (int32_t i = 0; i < CORPUS_SIZE; ++i) {
        int32_t situation = sit_dist(rng);
        auto emb = sensor.sample(situation, 0.05f);  // tight clustering
        std::memcpy(&graph.embeddings[size_t(i) * EMBEDDING_DIM],
                    emb.data(), EMBEDDING_DIM * sizeof(float));
        graph.modalities[i] = mod_dist(rng);  // mix of all three
        graph.timestamps[i] = float(i) * 0.001f;  // 1 ms ticks
    }

    graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
    std::fprintf(stderr, "Graph built: %d nodes, %d edges\n",
                 graph.num_nodes, graph.num_edges / 2);

    DeviceMemoryGraph dg = upload_to_device(graph);
    QueryContext ctx = create_query_context(CORPUS_SIZE, EMBEDDING_DIM, 5);

    // ── Retrieval config tuned for ultra-low-latency regime ──
    RetrievalConfig cfg;
    cfg.top_k             = 5;     // smaller K = lower top-K kernel cost
    cfg.bfs_max_hops      = 1;     // one hop is enough at 1 kHz
    cfg.time_decay_lambda = 1e-3f; // strong recency bias
    cfg.bfs_score_decay   = 0.5f;
    cfg.modality_filter   = -1;

    // ── Control loop ──
    int32_t relevant_hits = 0;
    int32_t total_queries = 0;

    for (int32_t tick = 0; tick < NUM_TICKS; ++tick) {
        int32_t situation = sit_dist(rng);
        auto query_emb = sensor.sample(situation, 0.06f);

        float query_ts = float(CORPUS_SIZE + tick) * 0.001f;

        RetrievalStats stats;
        auto results = query_memory_fast(dg, ctx, query_emb.data(), query_ts, cfg, stats);

        report.latencies.add(double(stats.gpu_ms_total));
        ++total_queries;

        // A "relevant hit" is any top-1 result with high cosine similarity,
        // indicating the retrieval found a memory from a similar situation.
        if (!results.empty() && results[0].score > 0.70f) ++relevant_hits;
    }

    report.num_frames = NUM_TICKS;
    report.add_metric("relevant_hit_rate",
                      double(relevant_hits) / total_queries);
    report.add_metric("total_queries", double(total_queries));
    report.add_metric("avg_queries_per_tick", 1.0);

    report.print_summary();
    report.print_json();

    destroy_query_context(ctx);
    free_device(dg);
    return report.passed() ? 0 : 1;
}
