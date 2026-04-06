// ============================================================================
//  tests/test_memory_graph.cpp
//
//  Host-only unit tests for the multimodal memory graph. No CUDA needed.
//  Verifies the structural invariants that the retrieval pipeline depends
//  on, so that a broken NSN construction is caught before it reaches the
//  GPU.
//
//  Tests:
//    T1  Synthetic corpus has correct per-modality counts
//    T2  Embeddings are approximately L2-normalized
//    T3  CSR format is internally consistent (row_offsets monotone,
//        col_indices in range, each row sorted)
//    T4  NSN is undirected (every edge a->b has a reciprocal b->a)
//    T5  No self-loops
//    T6  Cross-modal bridges: every node has at least one neighbor in
//        every other modality
//    T7  Average degree is within the expected bound
//
//  Build:  g++ -std=c++17 -Iinclude -o tests/run_tests \
//              src/memory_graph.cpp tests/test_memory_graph.cpp
//  Run:    ./tests/run_tests
// ============================================================================

#include "memory_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

// ─── Tiny assertion framework ───────────────────────────────────────
struct TestResult {
    std::string name;
    bool        passed;
    std::string message;
};

static std::vector<TestResult> g_results;

#define CHECK(expr, msg) do {                                                  \
    if (!(expr)) {                                                             \
        g_results.push_back({ __func__, false,                                 \
            std::string(#expr) + " failed: " + (msg) });                       \
        return;                                                                \
    }                                                                          \
} while (0)

#define PASS() do {                                                            \
    g_results.push_back({ __func__, true, "" });                               \
} while (0)

// ─── Tests ──────────────────────────────────────────────────────────

void T1_synthetic_counts() {
    auto g = MemoryGraph::synthetic(100, 50, 50, 768);
    CHECK(g.num_nodes == 200, "expected 200 total nodes");
    int counts[MOD_COUNT] = {0};
    for (int32_t m : g.modalities) {
        CHECK(m >= 0 && m < MOD_COUNT, "modality out of range");
        counts[m]++;
    }
    CHECK(counts[MOD_TEXT]  == 100, "wrong text count");
    CHECK(counts[MOD_AUDIO] == 50,  "wrong audio count");
    CHECK(counts[MOD_IMAGE] == 50,  "wrong image count");
    PASS();
}

void T2_embeddings_normalized() {
    auto g = MemoryGraph::synthetic(50, 25, 25, 256);
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        double norm = 0.0;
        for (int32_t d = 0; d < g.embedding_dim; ++d) {
            double v = g.embeddings[size_t(i) * g.embedding_dim + d];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        // L2 norms should be very close to 1.0 after normalization
        CHECK(std::fabs(norm - 1.0) < 1e-4, "embedding not L2-normalized");
    }
    PASS();
}

void T3_csr_format_valid() {
    auto g = MemoryGraph::synthetic(80, 40, 40, 128);
    g.build_nsn_edges();

    // Monotonicity
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        CHECK(g.row_offsets[i] <= g.row_offsets[i + 1],
              "row_offsets not monotone");
    }
    CHECK(g.row_offsets[g.num_nodes] == g.num_edges,
          "row_offsets[N] != num_edges");

    // In-range col_indices + sorted per row
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t prev = -1;
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            int32_t n = g.col_indices[j];
            CHECK(n >= 0 && n < g.num_nodes, "neighbor out of range");
            CHECK(n > prev, "neighbors not strictly sorted");
            prev = n;
        }
    }
    PASS();
}

void T4_edges_symmetric() {
    auto g = MemoryGraph::synthetic(60, 30, 30, 64);
    g.build_nsn_edges();

    // Build adjacency set for O(1) lookup
    std::vector<std::set<int32_t>> adj(g.num_nodes);
    for (int32_t i = 0; i < g.num_nodes; ++i)
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j)
            adj[i].insert(g.col_indices[j]);

    for (int32_t i = 0; i < g.num_nodes; ++i) {
        for (int32_t n : adj[i]) {
            CHECK(adj[n].count(i) == 1, "edge not reciprocated");
        }
    }
    PASS();
}

void T5_no_self_loops() {
    auto g = MemoryGraph::synthetic(50, 25, 25, 64);
    g.build_nsn_edges();
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            CHECK(g.col_indices[j] != i, "self-loop detected");
        }
    }
    PASS();
}

void T6_cross_modal_bridges() {
    auto g = MemoryGraph::synthetic(100, 50, 50, 64);
    g.build_nsn_edges();

    // For every node, check it has at least one neighbor in every OTHER
    // modality. This is the core guarantee the paper depends on.
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t my_mod = g.modalities[i];
        bool seen[MOD_COUNT] = { false };
        seen[my_mod] = true;  // trivially satisfied for own modality
        for (int32_t j = g.row_offsets[i]; j < g.row_offsets[i + 1]; ++j) {
            int32_t nb = g.col_indices[j];
            seen[g.modalities[nb]] = true;
        }
        for (int32_t m = 0; m < MOD_COUNT; ++m) {
            if (!seen[m]) {
                std::fprintf(stderr,
                    "  node %d (mod %d) has no neighbor of modality %d\n",
                    i, my_mod, m);
                CHECK(false, "missing cross-modal bridge");
            }
        }
    }
    PASS();
}

void T7_degree_bounded() {
    // Small enough that we can count: degree should be bounded by a
    // constant + log(N) from the NSN construction.
    auto g = MemoryGraph::synthetic(400, 200, 200, 64);
    g.build_nsn_edges();

    int32_t max_deg = 0;
    for (int32_t i = 0; i < g.num_nodes; ++i) {
        int32_t d = g.row_offsets[i + 1] - g.row_offsets[i];
        if (d > max_deg) max_deg = d;
    }
    double avg_deg = double(g.num_edges) / g.num_nodes;

    // Hub nodes get extra edges, so the max is higher, but the average
    // should be well under 30 for the NSN parameters we use.
    CHECK(avg_deg < 30.0, "average degree unexpectedly high");
    CHECK(max_deg < 200,  "max degree unexpectedly high");
    PASS();
}

// ─── Test runner ────────────────────────────────────────────────────

int main() {
    std::printf("\n═══════════════════════════════════════════════════\n");
    std::printf("  cuda-multimodal-memory unit tests\n");
    std::printf("═══════════════════════════════════════════════════\n\n");

    T1_synthetic_counts();
    T2_embeddings_normalized();
    T3_csr_format_valid();
    T4_edges_symmetric();
    T5_no_self_loops();
    T6_cross_modal_bridges();
    T7_degree_bounded();

    int32_t passed = 0, failed = 0;
    for (const auto& r : g_results) {
        if (r.passed) {
            std::printf("  ✓ %s\n", r.name.c_str());
            ++passed;
        } else {
            std::printf("  ✗ %s\n    %s\n", r.name.c_str(), r.message.c_str());
            ++failed;
        }
    }
    std::printf("\n  %d passed, %d failed\n\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
