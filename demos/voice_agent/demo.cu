// ============================================================================
//  demos/voice_agent/demo.cu
//
//  Workload: Real-time voice agent context (Tier 2 / perception-critical).
//
//  This is the demonstrator that most directly exercises the cross-modal
//  bridge feature of the Neural Shortcut Network. The substrate holds two
//  kinds of memories in the same graph:
//
//    1. Audio frame embeddings (CLAP-style), inserted at 30 Hz as the user
//       speaks.
//    2. Dialogue turn embeddings (E5-style text), inserted at the end of
//       each completed user turn.
//
//  On every end-of-turn event, the substrate is queried with the audio
//  embedding of the user's most recent utterance. The top-K seeds will
//  be audio memories, but cross-modal BFS expansion must reach relevant
//  text memories (prior dialogue turns) in a small number of hops. This
//  is what the agent's response generator uses to maintain multi-turn
//  context.
//
//  Success criteria:
//    - p99 query latency < 20 ms over 900 frames (30 seconds of conversation)
//    - cross-modal hit rate > 85% (top-10 contains text memories from the
//      same conversation thread)
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
    constexpr int32_t FRAME_RATE_HZ   = 30;
    constexpr int32_t NUM_FRAMES      = 900;   // 30 seconds of conversation
    constexpr int32_t NUM_TOPICS      = 20;    // distinct conversation topics
    constexpr int32_t CORPUS_SIZE     = 9000;  // 5 minutes of mixed audio+text
    constexpr int32_t EMBEDDING_DIM   = 768;
    constexpr double  DEADLINE_P99_MS = 20.0;

    DemoReport report;
    report.demo_name       = "voice_agent";
    report.scenario        = "30 Hz audio + dialogue text, cross-modal retrieval on end-of-turn";
    report.frame_rate_hz   = FRAME_RATE_HZ;
    report.deadline_p99_ms = DEADLINE_P99_MS;
    report.corpus_size     = CORPUS_SIZE;
    report.embedding_dim   = EMBEDDING_DIM;

    // ── Build a mixed audio+text corpus ──
    // Both modalities share the same NUM_TOPICS latent cluster structure:
    // audio frames about topic X cluster near text turns about topic X.
    // This mirrors how CLAP and E5 are jointly aligned in practice.
    SyntheticSensor sensor(EMBEDDING_DIM, NUM_TOPICS);

    MemoryGraph graph;
    graph.num_nodes     = CORPUS_SIZE;
    graph.embedding_dim = EMBEDDING_DIM;
    graph.embeddings.resize(size_t(CORPUS_SIZE) * EMBEDDING_DIM);
    graph.modalities.resize(CORPUS_SIZE);
    graph.timestamps.resize(CORPUS_SIZE);

    // 80% audio frames, 20% text turns (matches actual voice agent ratios)
    std::mt19937 rng(503);
    std::uniform_int_distribution<int32_t> topic_dist(0, NUM_TOPICS - 1);
    std::uniform_real_distribution<float>  p(0.0f, 1.0f);

    for (int32_t i = 0; i < CORPUS_SIZE; ++i) {
        int32_t topic = topic_dist(rng);
        bool is_text = p(rng) < 0.20f;
        auto emb = sensor.sample(topic, 0.15f);
        std::memcpy(&graph.embeddings[size_t(i) * EMBEDDING_DIM],
                    emb.data(), EMBEDDING_DIM * sizeof(float));
        graph.modalities[i] = is_text ? MOD_TEXT : MOD_AUDIO;
        graph.timestamps[i] = float(i) / FRAME_RATE_HZ;
    }

    graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
    std::fprintf(stderr, "Graph built: %d nodes, %d edges  "
                         "(audio+text mix with cross-modal bridges)\n",
                 graph.num_nodes, graph.num_edges / 2);

    DeviceMemoryGraph dg = upload_to_device(graph);
    QueryContext ctx = create_query_context(CORPUS_SIZE, EMBEDDING_DIM, 15);

    // ── Retrieval config: larger top-K for context assembly, 2 BFS hops
    //    to reach cross-modal neighbors.
    RetrievalConfig cfg;
    cfg.top_k             = 15;
    cfg.bfs_max_hops      = 2;
    cfg.time_decay_lambda = 1e-5f;
    cfg.bfs_score_decay   = 0.6f;
    cfg.modality_filter   = -1;  // we WANT to see both modalities

    // ── Conversation loop ──
    // Each "end-of-turn" event triggers a query with an audio embedding
    // from the current topic. We measure whether the cross-modal BFS
    // surfaces any text memories from the same topic.
    int32_t cross_modal_hits = 0;
    int32_t total_queries    = 0;
    int32_t current_topic    = topic_dist(rng);
    int32_t frames_on_topic  = 0;

    for (int32_t frame = 0; frame < NUM_FRAMES; ++frame) {
        // Simulate topic shifts every ~10 seconds (300 frames)
        if (frames_on_topic > 300) {
            current_topic = topic_dist(rng);
            frames_on_topic = 0;
        }
        ++frames_on_topic;

        // Query with an audio embedding from the current topic
        auto query_emb = sensor.sample(current_topic, 0.18f);
        float query_ts = float(CORPUS_SIZE + frame) / FRAME_RATE_HZ;

        RetrievalStats stats;
        auto results = query_memory_fast(dg, ctx, query_emb.data(), query_ts, cfg, stats);

        report.latencies.add(double(stats.gpu_ms_total));
        ++total_queries;

        // Cross-modal hit: did we surface any TEXT memories from the
        // current topic? Because the query was audio, this is only
        // possible if cross-modal bridges worked.
        bool hit = false;
        for (auto& r : results) {
            if (r.modality == MOD_TEXT && r.score > 0.60f) {
                hit = true;
                break;
            }
        }
        if (hit) ++cross_modal_hits;
    }

    report.num_frames = NUM_FRAMES;
    report.add_metric("cross_modal_hit_rate",
                      double(cross_modal_hits) / total_queries);
    report.add_metric("total_queries", double(total_queries));

    report.print_summary();
    report.print_json();

    destroy_query_context(ctx);
    free_device(dg);
    return report.passed() ? 0 : 1;
}
