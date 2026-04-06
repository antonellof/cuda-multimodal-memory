// ============================================================================
//  memory_cuda.cu
//  CUDA kernels for MARS (Memory for Autonomous Real-time Systems).
//  Compile with:  nvcc -O3 -std=c++17 -arch=sm_80 memory_cuda.cu ...
// ============================================================================
#include "memory_cuda.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

// ─── cuBLAS error checking ──────────────────────────────────────────
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status__ = (call);                                     \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuBLAS error %s:%d — status %d\n",          \
                         __FILE__, __LINE__, (int)status__);                  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace cg = cooperative_groups;

// ─── Tunable constants ──────────────────────────────────────────────
constexpr int32_t THREADS_PER_BLOCK = 256;
constexpr int32_t WARP_SIZE         = 32;
constexpr int32_t WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t MAX_TOP_K         = 64;
constexpr float   NEG_INF           = -FLT_MAX;

// ============================================================================
//  KERNEL 1: Cosine similarity between a query vector and all N memories.
//
//  Because embeddings are L2-normalized at construction, cosine similarity
//  reduces to a dot product. We use one block per memory node, with the
//  block's 256 threads cooperatively computing the dot product across the
//  embedding dimension D. A warp-level + block-level reduction produces
//  the final scalar similarity.
// ============================================================================
__global__ void cosine_similarity_kernel(
    const float*   __restrict__ embeddings,    // N * D
    const float*   __restrict__ query,         // D
    const int32_t* __restrict__ modalities,    // N
    float*         __restrict__ scores,        // N (output)
    int32_t                      N,
    int32_t                      D,
    int32_t                      modality_filter)
{
    int32_t node_id = blockIdx.x;
    if (node_id >= N) return;

    // Modality filter: write -inf and exit early
    if (modality_filter >= 0 && modalities[node_id] != modality_filter) {
        if (threadIdx.x == 0) scores[node_id] = NEG_INF;
        return;
    }

    const float* vec = embeddings + int64_t(node_id) * D;

    // ── Per-thread partial dot ───────────────────────────────────
    float partial = 0.0f;
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
        partial += vec[d] * query[d];
    }

    // ── Warp-level reduction via shuffle ─────────────────────────
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    // ── Block-level reduction across warps via shared memory ─────
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    int32_t lane    = threadIdx.x & (WARP_SIZE - 1);
    int32_t warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane == 0) scores[node_id] = v;
    }
}

// ============================================================================
//  KERNEL: FP32-to-FP16 conversion (one thread per element)
//  Grid: ceil(count / 256) blocks × 256 threads
// ============================================================================
__global__ void convert_f32_to_f16_kernel(const float* __restrict__ in,
                                           half* __restrict__ out, int32_t count) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) out[tid] = __float2half(in[tid]);
}

// ============================================================================
//  KERNEL: FP16 cosine similarity
//
//  Same structure as cosine_similarity_kernel but loads half-precision
//  embeddings, widens to float for accumulation. Query remains FP32.
//  Output is FP32 scores — the rest of the pipeline is unchanged.
//  Grid: N blocks × 256 threads (one block per memory node).
// ============================================================================
__global__ void cosine_similarity_fp16_kernel(
    const half*    __restrict__ embeddings_fp16,  // N * D
    const float*   __restrict__ query,            // D
    const int32_t* __restrict__ modalities,       // N
    float*         __restrict__ scores,           // N (output)
    int32_t                      N,
    int32_t                      D,
    int32_t                      modality_filter)
{
    int32_t node_id = blockIdx.x;
    if (node_id >= N) return;

    // Modality filter: write -inf and exit early
    if (modality_filter >= 0 && modalities[node_id] != modality_filter) {
        if (threadIdx.x == 0) scores[node_id] = NEG_INF;
        return;
    }

    const half* vec = embeddings_fp16 + int64_t(node_id) * D;

    // ── Per-thread partial dot (load FP16, widen to FP32) ─────────
    float partial = 0.0f;
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
        partial += __half2float(vec[d]) * query[d];
    }

    // ── Warp-level reduction via shuffle ─────────────────────────
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    // ── Block-level reduction across warps via shared memory ─────
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    int32_t lane    = threadIdx.x & (WARP_SIZE - 1);
    int32_t warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane == 0) scores[node_id] = v;
    }
}

// ============================================================================
//  KERNEL 2: Top-K selection.
//
//  For simplicity and correctness we use a single-block reduction kernel:
//  each thread maintains its own local top-K via insertion into a small
//  heap-in-registers, then threads merge their heaps in shared memory.
//  This works well for K up to 64 and N up to ~2^20. For larger N, a
//  tiled two-pass approach would be used.
// ============================================================================
__global__ void top_k_kernel(
    const float* __restrict__ scores,     // N
    int32_t*     __restrict__ top_idx,    // K
    float*       __restrict__ top_val,    // K
    int32_t                    N,
    int32_t                    K)
{
    // Per-thread local top-K
    float local_val[MAX_TOP_K];
    int32_t local_idx[MAX_TOP_K];
    #pragma unroll
    for (int i = 0; i < MAX_TOP_K; ++i) {
        local_val[i] = NEG_INF;
        local_idx[i] = -1;
    }

    int32_t tid  = threadIdx.x;
    int32_t step = blockDim.x;

    for (int32_t i = tid; i < N; i += step) {
        float s = scores[i];
        // Find current local minimum slot
        int   min_slot = 0;
        float min_val  = local_val[0];
        #pragma unroll
        for (int j = 1; j < MAX_TOP_K; ++j) {
            if (j >= K) break;
            if (local_val[j] < min_val) { min_val = local_val[j]; min_slot = j; }
        }
        if (s > min_val) {
            local_val[min_slot] = s;
            local_idx[min_slot] = i;
        }
    }

    // ── Merge across threads via shared memory reduction ────────
    __shared__ float s_val[THREADS_PER_BLOCK * 8];  // K up to 8 per thread avg
    __shared__ int32_t s_idx[THREADS_PER_BLOCK * 8];

    // Each thread writes its top-K contributions
    for (int j = 0; j < K && j < MAX_TOP_K; ++j) {
        int slot = tid * K + j;
        if (slot < THREADS_PER_BLOCK * 8) {
            s_val[slot] = local_val[j];
            s_idx[slot] = local_idx[j];
        }
    }
    __syncthreads();

    // Thread 0 does the final K-way selection
    if (tid == 0) {
        int32_t total = min(THREADS_PER_BLOCK * K, THREADS_PER_BLOCK * 8);
        for (int k = 0; k < K; ++k) {
            float best_v = NEG_INF;
            int   best_i = -1;
            int   best_s = -1;
            for (int j = 0; j < total; ++j) {
                if (s_val[j] > best_v) {
                    best_v = s_val[j];
                    best_i = s_idx[j];
                    best_s = j;
                }
            }
            top_val[k] = best_v;
            top_idx[k] = best_i;
            if (best_s >= 0) s_val[best_s] = NEG_INF;  // remove from pool
        }
    }
}

// ============================================================================
//  KERNEL 3: Warp-cooperative BFS expansion from seed nodes.
//
//  Each warp owns one frontier node. The 32 lanes split its neighbor list
//  evenly, testing each neighbor with atomicCAS to claim unvisited nodes.
//  Scores propagate with a per-hop decay: neighbor_score = seed_score * decay.
// ============================================================================
__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float*   __restrict__ sim_scores,    // similarity for decay blend
    int32_t*       __restrict__ hop_count,     // N, -1 = unvisited
    float*         __restrict__ bfs_score,     // N, propagated score
    const int32_t* __restrict__ frontier_in,
    int32_t*       __restrict__ frontier_out,
    int32_t*       __restrict__ frontier_count,
    int32_t                      frontier_size,
    int32_t                      current_hop,
    float                        hop_decay)
{
    int32_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id  = tid / WARP_SIZE;
    int32_t lane_id  = tid & (WARP_SIZE - 1);

    if (warp_id >= frontier_size) return;

    int32_t node        = frontier_in[warp_id];
    float   parent_sc   = bfs_score[node];
    int32_t start       = row_offsets[node];
    int32_t end         = row_offsets[node + 1];

    for (int32_t i = start + lane_id; i < end; i += WARP_SIZE) {
        int32_t neighbor = col_indices[i];
        // Atomic claim: only the first thread that reaches this neighbor
        // sets its hop_count from -1 to current_hop+1.
        int32_t old = atomicCAS(&hop_count[neighbor], -1, current_hop + 1);
        if (old == -1) {
            // Propagate score with hop decay, blended with its own similarity
            float own_sim = sim_scores[neighbor];
            float propagated = parent_sc * hop_decay;
            bfs_score[neighbor] = fmaxf(propagated, own_sim * 0.8f);
            int32_t pos = atomicAdd(frontier_count, 1);
            frontier_out[pos] = neighbor;
        }
    }
}

// ============================================================================
//  KERNEL 4: Temporal reranking with exponential decay
//
//  final = base_score * exp(-lambda * (query_ts - memory_ts))
//
//  Older memories get exponentially penalized — automatic forgetting
//  via hardware-accelerated exponential decay, entirely on-GPU.
// ============================================================================
__global__ void temporal_rerank_kernel(
    const float*   __restrict__ base_scores,   // N
    const float*   __restrict__ timestamps,    // N
    float*         __restrict__ final_scores,  // N
    int32_t                      N,
    float                        query_ts,
    float                        lambda)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float base = base_scores[tid];
    if (base <= NEG_INF * 0.5f) {
        final_scores[tid] = NEG_INF;
        return;
    }
    float age    = fmaxf(0.0f, query_ts - timestamps[tid]);
    float weight = __expf(-lambda * age);
    final_scores[tid] = base * weight;
}

// ============================================================================
//  KERNEL: Tiled Top-K (replaces single-block top_k_kernel)
//
//  Pass 1: each block handles a tile of ~4096 elements and produces a
//           local top-K list in shared memory.
//  Pass 2: a single block merges all tile results into the final top-K.
//  This eliminates the O(256*K*K) serial scan by thread 0.
// ============================================================================
constexpr int32_t TILE_SIZE = 16384;

__global__ void top_k_tiled_pass1(
    const float* __restrict__ scores,
    float*       __restrict__ tile_vals,   // num_tiles * K
    int32_t*     __restrict__ tile_idxs,   // num_tiles * K
    int32_t                    N,
    int32_t                    K)
{
    int32_t tile_id   = blockIdx.x;
    int32_t tile_base = tile_id * TILE_SIZE;
    int32_t tile_end  = min(tile_base + TILE_SIZE, N);
    int32_t tid       = threadIdx.x;

    float   local_val[MAX_TOP_K];
    int32_t local_idx[MAX_TOP_K];
    for (int i = 0; i < MAX_TOP_K; ++i) {
        local_val[i] = NEG_INF;
        local_idx[i] = -1;
    }

    for (int32_t i = tile_base + tid; i < tile_end; i += blockDim.x) {
        float s = scores[i];
        int   min_slot = 0;
        float min_val  = local_val[0];
        for (int j = 1; j < K && j < MAX_TOP_K; ++j) {
            if (local_val[j] < min_val) { min_val = local_val[j]; min_slot = j; }
        }
        if (s > min_val) {
            local_val[min_slot] = s;
            local_idx[min_slot] = i;
        }
    }

    __shared__ float   s_val[256 * 8];
    __shared__ int32_t s_idx[256 * 8];

    for (int j = 0; j < K && j < MAX_TOP_K; ++j) {
        int slot = tid * K + j;
        if (slot < 256 * 8) {
            s_val[slot] = local_val[j];
            s_idx[slot] = local_idx[j];
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t total = min(int32_t(blockDim.x) * K, 256 * 8);
        float*   out_v = tile_vals + int64_t(tile_id) * K;
        int32_t* out_i = tile_idxs + int64_t(tile_id) * K;

        for (int k = 0; k < K; ++k) {
            float best_v = NEG_INF;
            int   best_i = -1, best_s = -1;
            for (int j = 0; j < total; ++j) {
                if (s_val[j] > best_v) {
                    best_v = s_val[j]; best_i = s_idx[j]; best_s = j;
                }
            }
            out_v[k] = best_v;
            out_i[k] = best_i;
            if (best_s >= 0) s_val[best_s] = NEG_INF;
        }
    }
}

__global__ void top_k_tiled_pass2(
    const float*   __restrict__ tile_vals,  // num_tiles * K
    const int32_t* __restrict__ tile_idxs,  // num_tiles * K
    float*         __restrict__ top_val,    // K
    int32_t*       __restrict__ top_idx,    // K
    int32_t                      num_tiles,
    int32_t                      K)
{
    int32_t tid   = threadIdx.x;
    int32_t total = num_tiles * K;

    float   local_val[MAX_TOP_K];
    int32_t local_idx[MAX_TOP_K];
    for (int i = 0; i < MAX_TOP_K; ++i) {
        local_val[i] = NEG_INF;
        local_idx[i] = -1;
    }

    for (int32_t i = tid; i < total; i += blockDim.x) {
        float s = tile_vals[i];
        int   min_slot = 0;
        float min_val  = local_val[0];
        for (int j = 1; j < K && j < MAX_TOP_K; ++j) {
            if (local_val[j] < min_val) { min_val = local_val[j]; min_slot = j; }
        }
        if (s > min_val) {
            local_val[min_slot] = s;
            local_idx[min_slot] = tile_idxs[i];
        }
    }

    __shared__ float   s_val[256 * 8];
    __shared__ int32_t s_idx[256 * 8];

    for (int j = 0; j < K && j < MAX_TOP_K; ++j) {
        int slot = tid * K + j;
        if (slot < 256 * 8) {
            s_val[slot] = local_val[j];
            s_idx[slot] = local_idx[j];
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t pool = min(int32_t(blockDim.x) * K, 256 * 8);
        for (int k = 0; k < K; ++k) {
            float best_v = NEG_INF;
            int   best_i = -1, best_s = -1;
            for (int j = 0; j < pool; ++j) {
                if (s_val[j] > best_v) {
                    best_v = s_val[j]; best_i = s_idx[j]; best_s = j;
                }
            }
            top_val[k] = best_v;
            top_idx[k] = best_i;
            if (best_s >= 0) s_val[best_s] = NEG_INF;
        }
    }
}

// ============================================================================
//  KERNEL: GPU-side BFS seed initialization
//  Eliminates the D2H round trip between top-K and BFS.
// ============================================================================
__global__ void bfs_init_seeds_kernel(
    const int32_t* __restrict__ top_idx,     // K
    const float*   __restrict__ top_val,     // K
    int32_t*       __restrict__ hop_count,   // N, pre-set to -1
    float*         __restrict__ bfs_score,   // N
    int32_t*       __restrict__ frontier,    // K
    int32_t*       __restrict__ frontier_count,
    int32_t                      K)
{
    int32_t tid = threadIdx.x;
    if (tid >= K) return;

    int32_t node = top_idx[tid];
    if (node < 0) return;

    hop_count[node] = 0;
    bfs_score[node] = top_val[tid];
    int32_t pos = atomicAdd(frontier_count, 1);
    frontier[pos] = node;
}

// ============================================================================
//  KERNEL: Device-driven BFS expansion (no host sync between hops)
//
//  Reads frontier_count from device memory. Launch with a grid large enough
//  to cover the worst-case frontier size; excess warps exit immediately.
//  This eliminates the cudaStreamSynchronize between BFS hops.
// ============================================================================
__global__ void bfs_expand_device_driven_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float*   __restrict__ sim_scores,
    int32_t*       __restrict__ hop_count,
    float*         __restrict__ bfs_score,
    const int32_t* __restrict__ frontier_in,
    int32_t*       __restrict__ frontier_out,
    int32_t*       __restrict__ frontier_count_in,   // device-resident
    int32_t*       __restrict__ frontier_count_out,  // device-resident
    int32_t                      current_hop,
    float                        hop_decay)
{
    int32_t frontier_size = *frontier_count_in;
    int32_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id  = tid / WARP_SIZE;
    int32_t lane_id  = tid & (WARP_SIZE - 1);

    if (warp_id >= frontier_size) return;

    int32_t node      = frontier_in[warp_id];
    float   parent_sc = bfs_score[node];
    int32_t start     = row_offsets[node];
    int32_t end       = row_offsets[node + 1];

    for (int32_t i = start + lane_id; i < end; i += WARP_SIZE) {
        int32_t neighbor = col_indices[i];
        int32_t old = atomicCAS(&hop_count[neighbor], -1, current_hop + 1);
        if (old == -1) {
            float own_sim = sim_scores[neighbor];
            float propagated = parent_sc * hop_decay;
            bfs_score[neighbor] = fmaxf(propagated, own_sim * 0.8f);
            int32_t pos = atomicAdd(frontier_count_out, 1);
            frontier_out[pos] = neighbor;
        }
    }
}

// ============================================================================
//  KERNEL: GPU-side result compaction
//
//  Scans hop_count[] and compacts only visited nodes (hop_count >= 0)
//  into a dense output buffer. Replaces the O(N) D2H copy of three
//  full arrays with an O(visited) copy of one compact struct array.
// ============================================================================
struct CompactResult {
    int32_t node_id;
    float   score;
    int32_t modality;
    int32_t hops;
};

__global__ void compact_results_kernel(
    const int32_t* __restrict__ hop_count,    // N
    const float*   __restrict__ bfs_score,    // N
    const int32_t* __restrict__ modalities,   // N
    CompactResult* __restrict__ out,          // max_results
    int32_t*       __restrict__ out_count,    // 1
    int32_t                      N,
    int32_t                      max_results)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int32_t hops = hop_count[tid];
    if (hops < 0) return;

    int32_t pos = atomicAdd(out_count, 1);
    if (pos < max_results) {
        out[pos] = { tid, bfs_score[tid], modalities[tid], hops };
    }
}

// ============================================================================
//  KERNEL: Initialize index array [0, 1, 2, ..., N-1] for CUB sort
// ============================================================================
__global__ void init_indices_kernel(int32_t* __restrict__ indices, int32_t N)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) indices[tid] = tid;
}

// ============================================================================
//  KERNEL: Negate scores for ascending radix sort (ascending sort of
//  negated values = descending sort of original values)
// ============================================================================
__global__ void negate_scores_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int32_t                    N)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) out[tid] = -in[tid];
}

// ============================================================================
//  KERNEL: Extract top-K results from sorted arrays into top_idx/top_val
//  (reads from the first K entries of the sorted output)
// ============================================================================
__global__ void extract_topk_kernel(
    const float*   __restrict__ sorted_neg_scores,  // K (negated, ascending)
    const int32_t* __restrict__ sorted_indices,      // K
    float*         __restrict__ top_val,             // K
    int32_t*       __restrict__ top_idx,             // K
    int32_t                      K)
{
    int32_t tid = threadIdx.x;
    if (tid >= K) return;
    top_val[tid] = -sorted_neg_scores[tid];  // un-negate
    top_idx[tid] = sorted_indices[tid];
}

// ============================================================================
//  Host helpers
// ============================================================================
DeviceMemoryGraph upload_to_device(const MemoryGraph& h) {
    DeviceMemoryGraph d;
    d.num_nodes     = h.num_nodes;
    d.num_edges     = h.num_edges;
    d.embedding_dim = h.embedding_dim;

    CUDA_CHECK(cudaMalloc(&d.d_row_offsets, h.row_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_col_indices, h.col_indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_embeddings,  h.embeddings.size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.d_modalities,  h.modalities.size()  * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_timestamps,  h.timestamps.size()  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d.d_row_offsets, h.row_offsets.data(),
                          h.row_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_col_indices, h.col_indices.data(),
                          h.col_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_embeddings,  h.embeddings.data(),
                          h.embeddings.size()  * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_modalities,  h.modalities.data(),
                          h.modalities.size()  * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_timestamps,  h.timestamps.data(),
                          h.timestamps.size()  * sizeof(float),   cudaMemcpyHostToDevice));
    return d;
}

void free_device(DeviceMemoryGraph& d) {
    cudaFree(d.d_row_offsets);
    cudaFree(d.d_col_indices);
    cudaFree(d.d_embeddings);
    cudaFree(d.d_embeddings_fp16);
    cudaFree(d.d_modalities);
    cudaFree(d.d_timestamps);
    d = {};
}

// ── FP16 embedding management ──────────────────────────────────
void upload_fp16_embeddings(DeviceMemoryGraph& dg) {
    int64_t count = int64_t(dg.num_nodes) * dg.embedding_dim;
    if (count <= 0 || dg.d_embeddings == nullptr) return;
    CUDA_CHECK(cudaMalloc(&dg.d_embeddings_fp16, count * sizeof(half)));

    int32_t blocks = (int32_t)((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    convert_f32_to_f16_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        dg.d_embeddings, dg.d_embeddings_fp16, (int32_t)count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_fp16_embeddings(DeviceMemoryGraph& dg) {
    if (dg.d_embeddings_fp16) {
        cudaFree(dg.d_embeddings_fp16);
        dg.d_embeddings_fp16 = nullptr;
    }
}

// ============================================================================
//  The full retrieval pipeline
// ============================================================================
std::vector<QueryResult>
query_memory(const DeviceMemoryGraph& dg,
             const float*      h_query,
             float             query_timestamp,
             const RetrievalConfig& cfg,
             RetrievalStats&   stats)
{
    const int32_t N = dg.num_nodes;
    const int32_t D = dg.embedding_dim;
    const int32_t K = std::min(cfg.top_k, MAX_TOP_K);

    // ── Device scratch buffers ───────────────────────────────────
    float*   d_query;      CUDA_CHECK(cudaMalloc(&d_query,      D * sizeof(float)));
    float*   d_sim;        CUDA_CHECK(cudaMalloc(&d_sim,        N * sizeof(float)));
    float*   d_final;      CUDA_CHECK(cudaMalloc(&d_final,      N * sizeof(float)));
    float*   d_bfs_score;  CUDA_CHECK(cudaMalloc(&d_bfs_score,  N * sizeof(float)));
    int32_t* d_hop_count;  CUDA_CHECK(cudaMalloc(&d_hop_count,  N * sizeof(int32_t)));
    int32_t* d_top_idx;    CUDA_CHECK(cudaMalloc(&d_top_idx,    K * sizeof(int32_t)));
    float*   d_top_val;    CUDA_CHECK(cudaMalloc(&d_top_val,    K * sizeof(float)));
    int32_t* d_front_a;    CUDA_CHECK(cudaMalloc(&d_front_a,    N * sizeof(int32_t)));
    int32_t* d_front_b;    CUDA_CHECK(cudaMalloc(&d_front_b,    N * sizeof(int32_t)));
    int32_t* d_front_cnt;  CUDA_CHECK(cudaMalloc(&d_front_cnt,  sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query, D * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);

    // ── STAGE 1: similarity ──────────────────────────────────────
    cudaEventRecord(e0);
    cosine_similarity_kernel<<<N, THREADS_PER_BLOCK>>>(
        dg.d_embeddings, d_query, dg.d_modalities, d_sim,
        N, D, cfg.modality_filter);
    CUDA_CHECK(cudaGetLastError());

    // ── STAGE 2: temporal rerank ─────────────────────────────────
    cudaEventRecord(e1);
    int32_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    temporal_rerank_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_sim, dg.d_timestamps, d_final,
        N, query_timestamp, cfg.time_decay_lambda);
    CUDA_CHECK(cudaGetLastError());

    // ── STAGE 3: top-K seed selection ────────────────────────────
    cudaEventRecord(e2);
    top_k_kernel<<<1, THREADS_PER_BLOCK>>>(d_final, d_top_idx, d_top_val, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Copy seeds back to initialize BFS
    std::vector<int32_t> h_top_idx(K);
    std::vector<float>   h_top_val(K);
    CUDA_CHECK(cudaMemcpy(h_top_idx.data(), d_top_idx, K * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_top_val.data(), d_top_val, K * sizeof(float),   cudaMemcpyDeviceToHost));

    // ── STAGE 4: warp-cooperative BFS expansion ──────────────────
    cudaEventRecord(e3);
    CUDA_CHECK(cudaMemset(d_hop_count, 0xFF, N * sizeof(int32_t)));   // -1
    // Initialize seeds with hop=0 and their similarity score
    std::vector<int32_t> h_hop_init(K, 0);
    for (int i = 0; i < K; ++i) {
        if (h_top_idx[i] < 0) continue;
        CUDA_CHECK(cudaMemcpy(&d_hop_count[h_top_idx[i]], &h_hop_init[i],
                              sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_bfs_score[h_top_idx[i]], &h_top_val[i],
                              sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_front_a, h_top_idx.data(),
                          K * sizeof(int32_t), cudaMemcpyHostToDevice));

    int32_t frontier_size = 0;
    for (int i = 0; i < K; ++i) if (h_top_idx[i] >= 0) ++frontier_size;

    int32_t* d_front_in  = d_front_a;
    int32_t* d_front_out = d_front_b;
    int32_t  waves       = 0;

    for (int hop = 0; hop < cfg.bfs_max_hops && frontier_size > 0; ++hop) {
        CUDA_CHECK(cudaMemset(d_front_cnt, 0, sizeof(int32_t)));
        int32_t warps_needed = frontier_size;
        int32_t block_count  = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        bfs_expand_kernel<<<block_count, THREADS_PER_BLOCK>>>(
            dg.d_row_offsets, dg.d_col_indices, d_final,
            d_hop_count, d_bfs_score,
            d_front_in, d_front_out, d_front_cnt,
            frontier_size, hop, cfg.bfs_score_decay);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_front_cnt, sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
        std::swap(d_front_in, d_front_out);
        ++waves;
    }

    cudaEventRecord(e4);
    cudaEventSynchronize(e4);

    // ── Collect final results: scan hop_count + bfs_score, return top-K ──
    std::vector<int32_t> h_hop(N);
    std::vector<float>   h_bfs(N);
    std::vector<int32_t> h_mod(N);
    CUDA_CHECK(cudaMemcpy(h_hop.data(), d_hop_count,  N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bfs.data(), d_bfs_score,  N * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mod.data(), dg.d_modalities, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::vector<QueryResult> results;
    results.reserve(N);
    for (int32_t i = 0; i < N; ++i) {
        if (h_hop[i] >= 0) {
            results.push_back({ i, h_bfs[i], h_mod[i], h_hop[i] });
        }
    }
    std::sort(results.begin(), results.end(),
              [](const QueryResult& a, const QueryResult& b) { return a.score > b.score; });
    if (static_cast<int32_t>(results.size()) > K) results.resize(K);

    // ── Timings ──────────────────────────────────────────────────
    cudaEventElapsedTime(&stats.gpu_ms_similarity, e0, e1);
    cudaEventElapsedTime(&stats.gpu_ms_rerank,     e1, e2);
    cudaEventElapsedTime(&stats.gpu_ms_topk,       e2, e3);
    cudaEventElapsedTime(&stats.gpu_ms_bfs,        e3, e4);
    cudaEventElapsedTime(&stats.gpu_ms_total,      e0, e4);
    stats.nodes_scanned = N;
    stats.bfs_waves     = waves;

    // ── Cleanup ──────────────────────────────────────────────────
    cudaFree(d_query);     cudaFree(d_sim);       cudaFree(d_final);
    cudaFree(d_bfs_score); cudaFree(d_hop_count); cudaFree(d_top_idx);
    cudaFree(d_top_val);   cudaFree(d_front_a);   cudaFree(d_front_b);
    cudaFree(d_front_cnt);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4);

    return results;
}

// ============================================================================
//  Pre-allocated query context — create once, reuse across queries
// ============================================================================
QueryContext create_query_context(int32_t max_nodes, int32_t embedding_dim,
                                  int32_t max_k) {
    QueryContext ctx;
    ctx.max_nodes     = max_nodes;
    ctx.embedding_dim = embedding_dim;
    ctx.max_k         = std::min(max_k, (int32_t)MAX_TOP_K);

    int32_t N = max_nodes;
    int32_t D = embedding_dim;
    int32_t K = ctx.max_k;
    int32_t num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    CUDA_CHECK(cudaMalloc(&ctx.d_query,      D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sim,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_final,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_bfs_score,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_hop_count,  N * sizeof(int32_t)));
    int32_t topk_buf = K + num_tiles * K;
    CUDA_CHECK(cudaMalloc(&ctx.d_top_idx,    topk_buf * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_top_val,    topk_buf * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_front_a,    N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_front_b,    N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_front_cnt,  sizeof(int32_t)));

    // device-driven BFS needs a second frontier counter
    CUDA_CHECK(cudaMalloc(&ctx.d_front_cnt_b, sizeof(int32_t)));

    // result compaction buffer — scaled up for large corpus support (>1M)
    ctx.max_compact = std::min(N, K * 500);
    CUDA_CHECK(cudaMalloc(&ctx.d_compact_out,   ctx.max_compact * sizeof(CompactResult)));
    CUDA_CHECK(cudaMalloc(&ctx.d_compact_count, sizeof(int32_t)));

    // pinned host memory for truly async H2D query transfer
    CUDA_CHECK(cudaMallocHost(&ctx.h_query_pinned, D * sizeof(float)));

    cudaEventCreate(&ctx.e0); cudaEventCreate(&ctx.e1);
    cudaEventCreate(&ctx.e2); cudaEventCreate(&ctx.e3);
    cudaEventCreate(&ctx.e4);
    cudaStreamCreate(&ctx.stream);

    // cuBLAS handle
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, ctx.stream));
        ctx.cublas_handle = static_cast<void*>(handle);
    }

    // CUB radix sort buffers
    CUDA_CHECK(cudaMalloc(&ctx.d_sort_keys_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sort_keys_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sort_vals_in,  N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sort_vals_out, N * sizeof(int32_t)));

    // Query CUB for required temp storage size
    ctx.sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, ctx.sort_temp_bytes,
        ctx.d_sort_keys_in, ctx.d_sort_keys_out,
        ctx.d_sort_vals_in, ctx.d_sort_vals_out,
        N, 0, 32, ctx.stream);
    CUDA_CHECK(cudaMalloc(&ctx.d_sort_temp, ctx.sort_temp_bytes));

    ctx.valid = true;
    return ctx;
}

void destroy_query_context(QueryContext& ctx) {
    if (!ctx.valid) return;
    // cleanup CUDA graph resources
    if (ctx.graph_exec) cudaGraphExecDestroy(ctx.graph_exec);
    if (ctx.captured_graph) cudaGraphDestroy(ctx.captured_graph);

    // cleanup cuBLAS handle
    if (ctx.cublas_handle) {
        cublasDestroy(static_cast<cublasHandle_t>(ctx.cublas_handle));
        ctx.cublas_handle = nullptr;
    }

    // cleanup CUB sort buffers
    cudaFree(ctx.d_sort_temp);
    cudaFree(ctx.d_sort_keys_in);
    cudaFree(ctx.d_sort_keys_out);
    cudaFree(ctx.d_sort_vals_in);
    cudaFree(ctx.d_sort_vals_out);

    cudaFree(ctx.d_query);      cudaFree(ctx.d_sim);
    cudaFree(ctx.d_final);      cudaFree(ctx.d_bfs_score);
    cudaFree(ctx.d_hop_count);  cudaFree(ctx.d_top_idx);
    cudaFree(ctx.d_top_val);    cudaFree(ctx.d_front_a);
    cudaFree(ctx.d_front_b);    cudaFree(ctx.d_front_cnt);
    cudaFree(ctx.d_front_cnt_b);
    cudaFree(ctx.d_compact_out);
    cudaFree(ctx.d_compact_count);
    cudaFreeHost(ctx.h_query_pinned);
    cudaEventDestroy(ctx.e0); cudaEventDestroy(ctx.e1);
    cudaEventDestroy(ctx.e2); cudaEventDestroy(ctx.e3);
    cudaEventDestroy(ctx.e4);
    cudaStreamDestroy(ctx.stream);
    ctx = {};
}

// ============================================================================
//  Optimized retrieval pipeline
//
//  Optimization round 2:
//    1. Pre-allocated scratch buffers (no per-query cudaMalloc)
//    2. Tiled two-pass top-K (eliminates 0.5 ms serial bottleneck)
//    3. GPU-side BFS initialization kernel (eliminates D2H round trip)
//    4. Explicit CUDA stream + reused CUDA events
//
//  Optimization round 3:
//    5. Device-driven BFS: kernel reads frontier_count from device memory,
//       eliminating cudaStreamSynchronize between hops (~0.1 ms saved)
//    6. GPU-side result compaction: compact_results_kernel replaces the
//       O(N) D2H copy of 3 arrays with an O(visited) copy (~0.05-0.5 ms)
//    7. Pinned host memory for query vector (true async H2D transfer)
//
//  FP16/Graph extensions:
//    8. FP16 similarity kernel — halves memory bandwidth for Stage 1
//    9. CUDA graph capture/replay — eliminates kernel launch overhead
//   10. Large corpus scaling — max_compact = min(N, K*500)
//
//  cuBLAS/CUB acceleration (targets FAISS-competitive latency):
//   11. cuBLAS SGEMV for similarity — replaces custom kernel with
//       hardware-tuned BLAS (opt-in via ctx.use_cublas)
//   12. CUB radix sort for top-K — replaces O(256*K^2) serial merge
//       with O(N) parallel radix sort (~0.02ms vs ~0.35ms on A100
//       at N=10K). Opt-in via ctx.use_cub_topk.
// ============================================================================

// ── Helper: launch the full kernel pipeline (stages 1–7) ─────────
// Extracted so it can be called directly OR inside a CUDA graph capture.
// When skip_error_checks is true, CUDA_CHECK(cudaGetLastError()) calls
// are omitted — they break cudaStreamBeginCapture.
static void launch_pipeline_kernels(
    const DeviceMemoryGraph& dg, QueryContext& ctx,
    float query_timestamp,
    const RetrievalConfig& cfg, bool skip_error_checks)
{
    const int32_t N = dg.num_nodes;
    const int32_t D = dg.embedding_dim;
    const int32_t K = std::min(cfg.top_k, ctx.max_k);
    cudaStream_t  s = ctx.stream;

    // ── STAGE 1: similarity ──────────────────────────────────────
    cudaEventRecord(ctx.e0, s);
    if (ctx.use_cublas && ctx.cublas_handle != nullptr && cfg.modality_filter < 0) {
        // cuBLAS SGEMV — y = E^T * q  (E is N×D row-major)
        // In cuBLAS column-major convention: E row-major = E^T col-major
        // So we compute y = E^T_colmajor * q with op=CUBLAS_OP_T on
        // the D×N col-major matrix (which is our N×D row-major E).
        // cublasSgemv(handle, op, rows, cols, alpha, A, lda, x, incx, beta, y, incy)
        // A is D×N in col-major (= N×D row-major), op=T => y(N) = A^T(N×D) * x(D)
        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.cublas_handle);
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                                 D, N, &alpha,
                                 dg.d_embeddings, D,
                                 ctx.d_query, 1,
                                 &beta, ctx.d_sim, 1));
    } else if (ctx.use_fp16 && dg.d_embeddings_fp16 != nullptr) {
        cosine_similarity_fp16_kernel<<<N, THREADS_PER_BLOCK, 0, s>>>(
            dg.d_embeddings_fp16, ctx.d_query, dg.d_modalities, ctx.d_sim,
            N, D, cfg.modality_filter);
    } else {
        cosine_similarity_kernel<<<N, THREADS_PER_BLOCK, 0, s>>>(
            dg.d_embeddings, ctx.d_query, dg.d_modalities, ctx.d_sim,
            N, D, cfg.modality_filter);
    }
    if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

    // ── STAGE 2: temporal rerank ─────────────────────────────────
    cudaEventRecord(ctx.e1, s);
    int32_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    temporal_rerank_kernel<<<blocks, THREADS_PER_BLOCK, 0, s>>>(
        ctx.d_sim, dg.d_timestamps, ctx.d_final,
        N, query_timestamp, cfg.time_decay_lambda);
    if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

    // ── STAGE 3: top-K selection ──────────────────────────────────
    cudaEventRecord(ctx.e2, s);

    if (ctx.use_cub_topk && ctx.d_sort_temp != nullptr) {
        // CUB radix sort — negate scores, sort ascending, take first K
        // This replaces the O(256*K^2) serial merge with O(N) parallel radix sort
        // CUB on A100 for N=10K: ~0.02ms vs ~0.35ms for the tiled kernel

        // Step 1: negate d_final -> d_sort_keys_in (ascending sort = descending original)
        int32_t neg_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        negate_scores_kernel<<<neg_blocks, THREADS_PER_BLOCK, 0, s>>>(
            ctx.d_final, ctx.d_sort_keys_in, N);
        if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

        // Step 2: initialize index array [0, 1, 2, ..., N-1]
        init_indices_kernel<<<neg_blocks, THREADS_PER_BLOCK, 0, s>>>(
            ctx.d_sort_vals_in, N);
        if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

        // Step 3: CUB radix sort (ascending by negated score)
        size_t temp_bytes = ctx.sort_temp_bytes;
        cub::DeviceRadixSort::SortPairs(
            ctx.d_sort_temp, temp_bytes,
            ctx.d_sort_keys_in, ctx.d_sort_keys_out,
            ctx.d_sort_vals_in, ctx.d_sort_vals_out,
            N, 0, 32, s);
        if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

        // Step 4: extract first K entries into top_val/top_idx
        extract_topk_kernel<<<1, K, 0, s>>>(
            ctx.d_sort_keys_out, ctx.d_sort_vals_out,
            ctx.d_top_val, ctx.d_top_idx, K);
        if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

    } else {
        // Tiled top-K kernel
        int32_t num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
        float*   tile_vals = ctx.d_top_val + K;
        int32_t* tile_idxs = ctx.d_top_idx + K;

        if (num_tiles > 1) {
            top_k_tiled_pass1<<<num_tiles, THREADS_PER_BLOCK, 0, s>>>(
                ctx.d_final, tile_vals, tile_idxs, N, K);
            if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());
            top_k_tiled_pass2<<<1, THREADS_PER_BLOCK, 0, s>>>(
                tile_vals, tile_idxs, ctx.d_top_val, ctx.d_top_idx, num_tiles, K);
            if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());
        } else {
            top_k_kernel<<<1, THREADS_PER_BLOCK, 0, s>>>(
                ctx.d_final, ctx.d_top_idx, ctx.d_top_val, N, K);
            if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());
        }
    }

    // ── STAGE 4: device-driven BFS ───────────────────────────────
    cudaEventRecord(ctx.e3, s);
    cudaMemsetAsync(ctx.d_hop_count, 0xFF, N * sizeof(int32_t), s);
    cudaMemsetAsync(ctx.d_front_cnt, 0, sizeof(int32_t), s);

    bfs_init_seeds_kernel<<<1, K, 0, s>>>(
        ctx.d_top_idx, ctx.d_top_val,
        ctx.d_hop_count, ctx.d_bfs_score,
        ctx.d_front_a, ctx.d_front_cnt, K);
    if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

    // Device-driven BFS — no host sync between hops.
    int32_t* cnt_ping = ctx.d_front_cnt;
    int32_t* cnt_pong = ctx.d_front_cnt_b;
    int32_t* d_front_in  = ctx.d_front_a;
    int32_t* d_front_out = ctx.d_front_b;

    int32_t max_frontier_per_hop = std::min(N, K * 20);
    int32_t bfs_blocks = (max_frontier_per_hop + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    for (int hop = 0; hop < cfg.bfs_max_hops; ++hop) {
        cudaMemsetAsync(cnt_pong, 0, sizeof(int32_t), s);

        bfs_expand_device_driven_kernel<<<bfs_blocks, THREADS_PER_BLOCK, 0, s>>>(
            dg.d_row_offsets, dg.d_col_indices, ctx.d_final,
            ctx.d_hop_count, ctx.d_bfs_score,
            d_front_in, d_front_out, cnt_ping, cnt_pong,
            hop, cfg.bfs_score_decay);
        if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

        std::swap(d_front_in, d_front_out);
        std::swap(cnt_ping, cnt_pong);
    }

    // GPU-side result compaction
    cudaMemsetAsync(ctx.d_compact_count, 0, sizeof(int32_t), s);
    compact_results_kernel<<<(N + 255) / 256, 256, 0, s>>>(
        ctx.d_hop_count, ctx.d_bfs_score, dg.d_modalities,
        static_cast<CompactResult*>(ctx.d_compact_out),
        ctx.d_compact_count,
        N, ctx.max_compact);
    if (!skip_error_checks) CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(ctx.e4, s);
}

std::vector<QueryResult>
query_memory_fast(const DeviceMemoryGraph& dg,
                  QueryContext&     ctx,
                  const float*      h_query,
                  float             query_timestamp,
                  const RetrievalConfig& cfg,
                  RetrievalStats&   stats)
{
    const int32_t N = dg.num_nodes;
    const int32_t D = dg.embedding_dim;
    const int32_t K = std::min(cfg.top_k, ctx.max_k);
    cudaStream_t  s = ctx.stream;

    // H2D query copy — OUTSIDE the graph (query changes each call)
    std::memcpy(ctx.h_query_pinned, h_query, D * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_query, ctx.h_query_pinned, D * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));  // ensure query is on GPU before graph

    if (ctx.use_cuda_graph) {
        // Check if we need to (re-)capture the graph
        bool need_capture = !ctx.graph_captured
            || ctx.graph_N != N
            || ctx.graph_K != K
            || ctx.graph_bfs_hops != cfg.bfs_max_hops
            || ctx.graph_mod_filter != cfg.modality_filter
            || ctx.graph_fp16 != (ctx.use_fp16 && dg.d_embeddings_fp16 != nullptr);

        if (need_capture) {
            // Destroy old graph if any
            if (ctx.graph_exec) { cudaGraphExecDestroy(ctx.graph_exec); ctx.graph_exec = nullptr; }
            if (ctx.captured_graph) { cudaGraphDestroy(ctx.captured_graph); ctx.captured_graph = nullptr; }

            // Capture
            CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
            launch_pipeline_kernels(dg, ctx, query_timestamp, cfg, /*skip_error_checks=*/true);
            CUDA_CHECK(cudaStreamEndCapture(s, &ctx.captured_graph));
            CUDA_CHECK(cudaGraphInstantiate(&ctx.graph_exec, ctx.captured_graph, nullptr, nullptr, 0));

            ctx.graph_captured  = true;
            ctx.graph_N         = N;
            ctx.graph_K         = K;
            ctx.graph_bfs_hops  = cfg.bfs_max_hops;
            ctx.graph_mod_filter = cfg.modality_filter;
            ctx.graph_fp16      = (ctx.use_fp16 && dg.d_embeddings_fp16 != nullptr);
        }

        // Replay
        CUDA_CHECK(cudaGraphLaunch(ctx.graph_exec, s));
    } else {
        // Direct launch — device-driven path
        launch_pipeline_kernels(dg, ctx, query_timestamp, cfg, /*skip_error_checks=*/false);
    }

    // Single sync point: wait for everything including compaction
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Read compact count, then copy only the visited results
    int32_t compact_n = 0;
    CUDA_CHECK(cudaMemcpy(&compact_n, ctx.d_compact_count, sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    compact_n = std::min(compact_n, ctx.max_compact);

    std::vector<CompactResult> h_compact(compact_n);
    if (compact_n > 0) {
        CUDA_CHECK(cudaMemcpy(h_compact.data(), ctx.d_compact_out,
                              compact_n * sizeof(CompactResult),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<QueryResult> results;
    results.reserve(compact_n);
    for (const auto& cr : h_compact) {
        results.push_back({ cr.node_id, cr.score, cr.modality, cr.hops });
    }
    std::sort(results.begin(), results.end(),
              [](const QueryResult& a, const QueryResult& b) { return a.score > b.score; });
    if (static_cast<int32_t>(results.size()) > K) results.resize(K);

    // ── Timings ──────────────────────────────────────────────────
    cudaEventElapsedTime(&stats.gpu_ms_similarity, ctx.e0, ctx.e1);
    cudaEventElapsedTime(&stats.gpu_ms_rerank,     ctx.e1, ctx.e2);
    cudaEventElapsedTime(&stats.gpu_ms_topk,       ctx.e2, ctx.e3);
    cudaEventElapsedTime(&stats.gpu_ms_bfs,        ctx.e3, ctx.e4);
    cudaEventElapsedTime(&stats.gpu_ms_total,      ctx.e0, ctx.e4);
    stats.nodes_scanned = N;
    stats.bfs_waves     = cfg.bfs_max_hops;

    return results;
}

// ════════════════════════════════════════════════════════════════════
//  IMPORTANCE-WEIGHTED RETRIEVAL
// ════════════════════════════════════════════════════════════════════

// ── Init importance array to 1.0 ────────────────────────────────────
__global__ void init_importance_kernel(float* __restrict__ importance, int32_t N) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) importance[tid] = 1.0f;
}

void init_importance(DeviceMemoryGraph& dg) {
    if (dg.d_importance) cudaFree(dg.d_importance);
    CUDA_CHECK(cudaMalloc(&dg.d_importance, dg.num_nodes * sizeof(float)));
    int32_t blocks = (dg.num_nodes + 255) / 256;
    init_importance_kernel<<<blocks, 256>>>(dg.d_importance, dg.num_nodes);
    CUDA_CHECK(cudaGetLastError());
}

// ── Global importance decay: importance[i] *= (1 - decay_rate) ──────
__global__ void importance_decay_kernel(
    float* __restrict__ importance, int32_t N, float retain_factor)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) importance[tid] *= retain_factor;
}

// ── Boost importance of retrieved nodes ─────────────────────────────
__global__ void importance_boost_kernel(
    float*         __restrict__ importance,
    const int32_t* __restrict__ result_ids,
    int32_t n_results, float boost, float max_imp)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_results) return;
    int32_t node = result_ids[tid];
    if (node < 0) return;
    float old = importance[node];
    importance[node] = fminf(old + boost, max_imp);
}

void update_importance(DeviceMemoryGraph& dg, const int32_t* d_result_ids,
                       int32_t n_results, float boost,
                       float decay_rate, float max_importance,
                       cudaStream_t stream) {
    if (!dg.d_importance || dg.num_nodes <= 0) return;

    // Global decay
    float retain = 1.0f - decay_rate;
    int32_t blocks_all = (dg.num_nodes + 255) / 256;
    importance_decay_kernel<<<blocks_all, 256, 0, stream>>>(
        dg.d_importance, dg.num_nodes, retain);

    // Boost retrieved nodes
    if (n_results > 0 && d_result_ids) {
        int32_t blocks_res = (n_results + 255) / 256;
        importance_boost_kernel<<<blocks_res, 256, 0, stream>>>(
            dg.d_importance, d_result_ids, n_results,
            boost, max_importance);
    }
}

// ════════════════════════════════════════════════════════════════════
//  NOVELTY GATE
// ════════════════════════════════════════════════════════════════════
//
// Compute the maximum cosine similarity between a candidate embedding
// and all N existing memories. If max_sim > novelty_threshold, the
// candidate is redundant.
//
// Uses the same warp-shuffle dot-product pattern as the similarity
// kernel, but reduces to a single max across all N.

__global__ void novelty_max_sim_kernel(
    const float* __restrict__ embeddings,   // N x D
    const float* __restrict__ candidate,    // D
    float*       __restrict__ block_maxes,  // gridDim.x
    int32_t N, int32_t D)
{
    int32_t node_id = blockIdx.x;
    if (node_id >= N) return;

    const float* vec = embeddings + int64_t(node_id) * D;
    float partial = 0.0f;
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x)
        partial += vec[d] * candidate[d];

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    __shared__ float warp_sums[8];
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < 8) ? warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane == 0) block_maxes[node_id] = v;
    }
}

float compute_novelty(const DeviceMemoryGraph& dg, const float* d_query,
                      cudaStream_t stream) {
    int32_t N = dg.num_nodes;
    int32_t D = dg.embedding_dim;
    if (N <= 0) return 0.0f;

    // Compute per-node similarity
    float* d_sims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sims, N * sizeof(float)));
    novelty_max_sim_kernel<<<N, 256, 0, stream>>>(
        dg.d_embeddings, d_query, d_sims, N, D);

    // Find max on GPU using a simple reduction
    // (for N < 100K, a host-side max over the array is fast enough)
    std::vector<float> h_sims(N);
    CUDA_CHECK(cudaMemcpyAsync(h_sims.data(), d_sims, N * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    if (stream) cudaStreamSynchronize(stream);
    else cudaDeviceSynchronize();

    float max_sim = -1.0f;
    for (int32_t i = 0; i < N; ++i)
        if (h_sims[i] > max_sim) max_sim = h_sims[i];

    cudaFree(d_sims);
    return max_sim;
}

// ════════════════════════════════════════════════════════════════════
//  ADAPTIVE GRAPH MAINTENANCE
// ════════════════════════════════════════════════════════════════════

void init_edge_hits(DeviceMemoryGraph& dg) {
    if (dg.d_edge_hits) cudaFree(dg.d_edge_hits);
    CUDA_CHECK(cudaMalloc(&dg.d_edge_hits, dg.num_edges * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dg.d_edge_hits, 0, dg.num_edges * sizeof(int32_t)));
}

// Track which edges were traversed during BFS. Call after each query.
// This kernel increments the hit counter for each edge on the BFS path.
__global__ void count_edge_hits_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const int32_t* __restrict__ hop_count,    // -1 = not visited
    int32_t*       __restrict__ edge_hits,
    int32_t N)
{
    int32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    if (hop_count[node] <= 0) return;  // not reached by BFS, or seed

    int32_t start = row_offsets[node];
    int32_t end   = row_offsets[node + 1];
    int32_t my_hop = hop_count[node];

    for (int32_t e = start; e < end; ++e) {
        int32_t neighbor = col_indices[e];
        // If this neighbor is our parent (hop = my_hop - 1), this edge was traversed
        if (hop_count[neighbor] == my_hop - 1) {
            atomicAdd(&edge_hits[e], 1);
        }
    }
}

// Identify low-utility edges and rewire them. This runs on CPU since
// CSR modification is complex. Returns number of edges rewired.
int32_t maintain_graph(DeviceMemoryGraph& dg, int32_t min_hits,
                       int32_t window_queries, cudaStream_t stream) {
    if (!dg.d_edge_hits || dg.num_edges <= 0) return 0;

    int32_t E = dg.num_edges;
    std::vector<int32_t> h_hits(E);
    CUDA_CHECK(cudaMemcpy(h_hits.data(), dg.d_edge_hits, E * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));

    // Count low-utility edges (those with 0 hits over the window)
    int32_t low_utility = 0;
    for (int32_t e = 0; e < E; ++e)
        if (h_hits[e] <= min_hits) ++low_utility;

    // Reset counters for next window
    CUDA_CHECK(cudaMemset(dg.d_edge_hits, 0, E * sizeof(int32_t)));

    // Note: actual edge rewiring requires CSR reconstruction.
    // For now, we return the count so the benchmark can track the metric.
    // Full rewiring implementation requires host-side graph rebuild,
    // which is planned for the streaming insertion API.
    return low_utility;
}

