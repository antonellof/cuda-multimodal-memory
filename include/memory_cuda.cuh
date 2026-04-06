// ============================================================================
//  memory_cuda.cuh
//  GPU kernels for MARS.
//
//  The retrieval pipeline has four stages, executed in this order:
//
//    (1) cosine_similarity_kernel (FP32) or cosine_similarity_fp16_kernel (FP16)
//        Computes similarity between query embedding and all N memories
//        in parallel. One block per memory, warp-shuffle reduction inside.
//        FP16 variant loads half-precision embeddings, widens to float
//        for accumulation. Halves memory bandwidth for the dominant kernel.
//
//    (2) temporal_rerank_kernel
//        Applies exponential time decay: score = sim * exp(-lambda * age)
//        so recent memories outrank stale ones of equal similarity.
//
//    (3) top_k_kernel
//        Selects the top-K highest-scoring memories using per-thread
//        register heaps merged through shared memory in a single block.
//
//    (4) bfs_expand_kernel
//        Warp-cooperative BFS from the top-K seeds along NSN edges to
//        surface semantically related memories reachable in few hops.
//        This is what makes the system fast — instead of scanning all N
//        memories, we only "light up" the neighborhood of hot seeds.
//
//  FP16 support:
//    - FP16 similarity kernel (opt-in via ctx.use_fp16)
//    - CUDA graph capture/replay (opt-in via ctx.use_cuda_graph)
//    - Large corpus scaling (max_compact = min(N, K*500))
//
//  cuBLAS/CUB acceleration:
//    - cuBLAS SGEMV for similarity (opt-in via ctx.use_cublas)
//    - CUB radix sort for top-K (opt-in via ctx.use_cub_topk)
//    Both eliminate the top-K bottleneck (~0.35ms -> ~0.02ms)
// ============================================================================
#ifndef MEMORY_CUDA_CUH
#define MEMORY_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "memory_graph.h"

// ─── Error checking macro ───────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d — %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err__));    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ─── Device-resident graph (just pointers into GPU memory) ──────────
struct DeviceMemoryGraph {
    int32_t  num_nodes     = 0;
    int32_t  num_edges     = 0;
    int32_t  embedding_dim = 0;
    int32_t* d_row_offsets = nullptr;
    int32_t* d_col_indices = nullptr;
    float*   d_embeddings  = nullptr;  // N * D, row-major, L2-normalized
    half*    d_embeddings_fp16 = nullptr;  // N * D, FP16 copy
    int32_t* d_modalities  = nullptr;
    float*   d_timestamps  = nullptr;

    // importance scoring — per-node importance weight, updated on access
    float*   d_importance   = nullptr;  // N, initialized to 1.0
    // edge traversal counters — for adaptive graph maintenance
    int32_t* d_edge_hits    = nullptr;  // E, incremented on BFS traversal
};

DeviceMemoryGraph upload_to_device(const MemoryGraph& h);
void              free_device(DeviceMemoryGraph& d);

// FP16 embedding management — converts existing FP32 embeddings on GPU
void upload_fp16_embeddings(DeviceMemoryGraph& dg);
void free_fp16_embeddings(DeviceMemoryGraph& dg);

// ─── Query + retrieval API ──────────────────────────────────────────
struct QueryResult {
    int32_t node_id;
    float   score;
    int32_t modality;
    int32_t hops_from_seed;   // 0 = direct similarity hit, 1+ = BFS neighbor
};

struct RetrievalConfig {
    int32_t top_k          = 10;
    int32_t bfs_max_hops   = 2;     // expand seeds this many hops through NSN
    float   time_decay_lambda = 1e-8f;  // per-second decay rate
    float   bfs_score_decay   = 0.5f;   // per-hop score multiplier
    int32_t modality_filter   = -1;     // -1 = any
    bool    use_importance    = false;   // multiply by importance weight
};

struct RetrievalStats {
    float gpu_ms_similarity = 0.0f;
    float gpu_ms_topk       = 0.0f;
    float gpu_ms_bfs        = 0.0f;
    float gpu_ms_rerank     = 0.0f;
    float gpu_ms_total      = 0.0f;
    int32_t nodes_scanned   = 0;
    int32_t bfs_waves       = 0;
};

// Main entry: given a D-dimensional query embedding and a query timestamp
// (for temporal decay), return the top-K ranked memories.
std::vector<QueryResult>
query_memory(const DeviceMemoryGraph& dg,
             const float*      h_query,         // length = embedding_dim
             float             query_timestamp,
             const RetrievalConfig& cfg,
             RetrievalStats&   stats);

// ─── Pre-allocated query context (eliminates per-query cudaMalloc) ───
// Create once, reuse across all queries to the same graph.
struct QueryContext {
    int32_t max_nodes     = 0;
    int32_t embedding_dim = 0;
    int32_t max_k         = 0;

    float*   d_query      = nullptr;
    float*   d_sim        = nullptr;
    float*   d_final      = nullptr;
    float*   d_bfs_score  = nullptr;
    int32_t* d_hop_count  = nullptr;
    int32_t* d_top_idx    = nullptr;
    float*   d_top_val    = nullptr;
    int32_t* d_front_a    = nullptr;
    int32_t* d_front_b    = nullptr;
    int32_t* d_front_cnt  = nullptr;

    // result compaction buffers
    void*    d_compact_out   = nullptr;  // CompactResult array
    int32_t* d_compact_count = nullptr;
    int32_t  max_compact     = 0;

    // second frontier counter for device-driven BFS
    int32_t* d_front_cnt_b   = nullptr;

    // pinned host memory for query vector (enables true async H2D)
    float*   h_query_pinned  = nullptr;

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaStream_t stream    = nullptr;
    bool valid             = false;

    // CUDA graph capture/replay
    cudaGraph_t      captured_graph  = nullptr;
    cudaGraphExec_t  graph_exec      = nullptr;
    bool             graph_captured  = false;
    int32_t          graph_N         = 0;   // N at capture time
    int32_t          graph_K         = 0;
    int32_t          graph_bfs_hops  = 0;
    int32_t          graph_mod_filter = -1;
    bool             graph_fp16      = false;

    // feature flags (opt-in)
    bool use_cuda_graph = false;
    bool use_fp16       = false;

    // cuBLAS similarity + CUB radix-sort top-K
    bool use_cublas     = false;  // replace cosine_similarity_kernel with cublasSgemv
    bool use_cub_topk   = false;  // replace top_k_kernel with cub::DeviceRadixSort

    // cuBLAS handle (void* to avoid #include <cublas_v2.h> in header)
    void* cublas_handle = nullptr;

    // CUB radix sort buffers
    void*    d_sort_temp       = nullptr;   // CUB temp storage
    size_t   sort_temp_bytes   = 0;         // size of temp storage
    float*   d_sort_keys_in    = nullptr;   // negated scores (sort ascending = top-K descending)
    float*   d_sort_keys_out   = nullptr;   // sorted negated scores
    int32_t* d_sort_vals_in    = nullptr;   // index array [0..N-1]
    int32_t* d_sort_vals_out   = nullptr;   // sorted indices
};

QueryContext create_query_context(int32_t max_nodes, int32_t embedding_dim,
                                  int32_t max_k = 64);
void         destroy_query_context(QueryContext& ctx);

// Optimized query using pre-allocated context (no per-query malloc).
std::vector<QueryResult>
query_memory_fast(const DeviceMemoryGraph& dg,
                  QueryContext&     ctx,
                  const float*      h_query,
                  float             query_timestamp,
                  const RetrievalConfig& cfg,
                  RetrievalStats&   stats);

// ═══════════════════════════════════════════════════════════════════
//  Importance-Weighted Retrieval + Novelty Gate + Adaptive Graph
// ═══════════════════════════════════════════════════════════════════

// ─── Importance scoring ─────────────────────────────────────────────
// Initialize importance array to 1.0 for all nodes. Called once after
// upload_to_device().
void init_importance(DeviceMemoryGraph& dg);

// After each query, boost importance of returned result nodes.
// importance[node] = min(importance[node] + boost, max_importance)
// Also apply global decay: importance[i] *= (1 - decay_rate) for all i.
// This runs asynchronously on the query stream.
void update_importance(DeviceMemoryGraph& dg, const int32_t* d_result_ids,
                       int32_t n_results, float boost = 0.1f,
                       float decay_rate = 0.001f, float max_importance = 5.0f,
                       cudaStream_t stream = nullptr);

// ─── Novelty gate ───────────────────────────────────────────────────
// Before inserting a new memory, compute its maximum cosine similarity
// against all existing N memories. If max_sim > threshold, the memory
// is redundant and should be rejected (or merged).
// Returns the max similarity (host-side). Runs synchronously.
float compute_novelty(const DeviceMemoryGraph& dg, const float* d_query,
                      cudaStream_t stream = nullptr);

// ─── Adaptive graph maintenance ─────────────────────────────────────
// Initialize edge hit counters to zero. Called once after upload.
void init_edge_hits(DeviceMemoryGraph& dg);

// Run a maintenance pass: identify low-utility edges (hit_count below
// threshold over the last window) and rewire them to connect nodes that
// were co-retrieved but lack direct edges. Call periodically (e.g.,
// every 1000 queries) during idle time between sensor frames.
// Returns number of edges rewired.
int32_t maintain_graph(DeviceMemoryGraph& dg, int32_t min_hits = 0,
                       int32_t window_queries = 1000,
                       cudaStream_t stream = nullptr);

#endif // MEMORY_CUDA_CUH
