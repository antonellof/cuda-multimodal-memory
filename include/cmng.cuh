// ============================================================================
//  cmng.cuh — Cross-Modal Navigable Graph (CMNG)
//
//  A CAGRA-inspired fixed-degree directed graph with cross-modal bridges
//  and temporal decay. Replaces the O(N·D) brute-force similarity scan with
//  O(beam_width · degree · iterations · D) greedy beam search.
//
//  Key properties:
//    - Fixed out-degree d for all nodes (GPU-friendly uniform computation)
//    - Last d_cross slots per node are cross-modal bridges (O(1) reachability)
//    - Stored as flat int32_t[N * d] array (coalesced access, not CSR)
//    - Greedy beam search with warp-cooperative distance computation
//    - Temporal decay applied to candidate set (O(visited), not O(N))
//
//  Construction: tiled brute-force k-NN via cuBLAS SGEMM + cross-modal bridges
//  Search: single-CTA beam search with shared-memory hash table
// ============================================================================
#ifndef CMNG_CUH
#define CMNG_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// Reuse QueryResult and RetrievalStats from existing API
#include "memory_cuda.cuh"

// ─── Error checking (reuse from memory_cuda.cuh) ─────────────────────
#ifndef CMNG_CHECK
#define CMNG_CHECK(call) CUDA_CHECK(call)
#endif

// ─── CMNG Graph Structure ────────────────────────────────────────────
struct CMNGGraph {
    int32_t N            = 0;       // number of nodes
    int32_t D            = 0;       // embedding dimension
    int32_t degree       = 64;      // fixed out-degree per node
    int32_t degree_cross = 4;       // cross-modal bridge slots (last d_cross edges)

    // Device arrays
    int32_t* d_neighbors       = nullptr;  // N * degree (flat 2D: node v's edges at [v*degree .. (v+1)*degree-1])
    float*   d_embeddings      = nullptr;  // N * D, L2-normalized, row-major
    half*    d_embeddings_fp16 = nullptr;  // N * D, FP16 copy (optional)
    int32_t* d_modalities      = nullptr;  // N
    float*   d_timestamps      = nullptr;  // N

    // Construction metadata
    float    build_time_ms     = 0.0f;     // total construction time
};

// ─── Search Configuration ────────────────────────────────────────────
struct CMNGSearchConfig {
    int32_t K               = 10;      // results to return
    int32_t beam_width      = 128;     // internal top-M list size (>= K)
    int32_t max_iterations  = 64;      // max search iterations before forced stop
    int32_t num_parents     = 1;       // p: nodes expanded per iteration
    float   time_decay_lambda = 1e-8f; // temporal decay rate
    int32_t modality_filter = -1;      // -1 = any modality in results
    bool    use_fp16        = false;    // use FP16 embeddings for distance
};

// ─── Pre-allocated Search Context ────────────────────────────────────
// Create once, reuse across queries (same pattern as QueryContext).
struct CMNGSearchContext {
    int32_t max_D         = 0;
    int32_t max_K         = 0;
    int32_t max_beam      = 0;

    // Device buffers for results
    int32_t* d_result_ids    = nullptr;   // K
    float*   d_result_scores = nullptr;   // K
    int32_t* d_result_mods   = nullptr;   // K
    int32_t* d_result_count  = nullptr;   // 1

    // Pinned host memory for query
    float*   h_query_pinned  = nullptr;   // D
    float*   d_query         = nullptr;   // D

    cudaEvent_t e_start, e_end;
    cudaStream_t stream      = nullptr;
    bool valid               = false;
};

// ─── Construction API ────────────────────────────────────────────────

// Build a CMNG from host-side data. Uploads embeddings, builds k-NN
// graph via tiled cuBLAS SGEMM, inserts cross-modal bridges.
// Construction is a one-time cost; the returned graph is ready for search.
CMNGGraph build_cmng(const float*   h_embeddings,    // N * D, L2-normalized
                     const int32_t* h_modalities,    // N
                     const float*   h_timestamps,    // N
                     int32_t N, int32_t D,
                     int32_t degree       = 64,
                     int32_t degree_cross = 4);

// Upload FP16 copy of embeddings (call after build_cmng)
void cmng_upload_fp16(CMNGGraph& g);

// Free all device memory
void free_cmng(CMNGGraph& g);

// ─── Search API ──────────────────────────────────────────────────────

CMNGSearchContext create_cmng_context(int32_t D, int32_t max_beam = 128,
                                      int32_t max_k = 64);
void destroy_cmng_context(CMNGSearchContext& ctx);

// Single-query greedy beam search with temporal decay.
// Returns top-K results sorted by time-decayed score.
std::vector<QueryResult>
cmng_search(const CMNGGraph&      g,
            CMNGSearchContext&     ctx,
            const float*          h_query,        // D, L2-normalized
            float                 query_timestamp,
            const CMNGSearchConfig& cfg,
            RetrievalStats&       stats);

#endif // CMNG_CUH
