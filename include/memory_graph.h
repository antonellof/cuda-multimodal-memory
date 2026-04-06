// ============================================================================
//  memory_graph.h
//  GPU-resident multimodal memory graph. Each node represents a "memory"
//  — a text snippet, an audio clip, or an image — with:
//    • a dense embedding vector (D dimensions, typically 768 or 1024)
//    • a modality tag  (TEXT | AUDIO | IMAGE)
//    • a timestamp     (for temporal decay / "forgetting")
//    • CSR-format edges to other memories in the Neural Shortcut Network
// ============================================================================
#ifndef MEMORY_GRAPH_H
#define MEMORY_GRAPH_H

#include <cstdint>
#include <string>
#include <vector>

enum Modality : int32_t {
    MOD_TEXT  = 0,
    MOD_AUDIO = 1,
    MOD_IMAGE = 2,
    MOD_COUNT = 3
};

// Host-side representation of the multimodal memory graph
struct MemoryGraph {
    int32_t num_nodes      = 0;
    int32_t num_edges      = 0;        // edges counted twice (undirected)
    int32_t embedding_dim  = 0;        // D

    // CSR topology
    std::vector<int32_t> row_offsets;  // (N+1)
    std::vector<int32_t> col_indices;  // num_edges

    // Per-node payload
    std::vector<float>   embeddings;   // N * D, row-major
    std::vector<int32_t> modalities;   // N
    std::vector<float>   timestamps;   // N  (seconds since epoch, float ok here)

    // Build a small synthetic multimodal corpus for testing
    static MemoryGraph synthetic(int32_t n_text,
                                 int32_t n_audio,
                                 int32_t n_image,
                                 int32_t dim,
                                 uint32_t seed = 42);

    // Build Neural Shortcut Network edges over the existing nodes
    void build_nsn_edges(int32_t k = 6, double p = 0.15);

    // Phase-disable flags for ablation studies
    struct NSNConfig {
        int32_t k              = 6;
        double  p              = 0.15;
        bool    enable_ring    = true;   // Phase 1: ring lattice
        bool    enable_skips   = true;   // Phase 2: hierarchical skip connections
        bool    enable_hubs    = true;   // Phase 3: hub supernodes at sqrt(N)
        bool    enable_rewire  = true;   // Phase 4: small-world rewiring
        bool    enable_bridges = true;   // Phase 5: cross-modal bridges
    };

    // Build NSN with configurable phase enables (for ablation experiments)
    void build_nsn_edges_configurable(const NSNConfig& cfg);

    void print_summary() const;
    size_t device_bytes() const;
};

#endif // MEMORY_GRAPH_H
