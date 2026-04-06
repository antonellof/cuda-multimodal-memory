// ============================================================================
//  cmng_search.cu — Greedy beam search on the Cross-Modal Navigable Graph
//
//  Single-CTA beam search: one thread block per query, 256 threads (8 warps).
//  Each warp cooperatively computes one distance (D/32 elements per thread).
//  Visited tracking via open-addressing hash table in shared memory.
//  Temporal decay applied to the final candidate set.
//
//  Complexity: O(max_iterations * degree * D / 32) per query — independent of N.
// ============================================================================
#include "cmng.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// ─── Constants ───────────────────────────────────────────────────────
constexpr int32_t SEARCH_THREADS  = 256;
constexpr int32_t SEARCH_WARPS    = SEARCH_THREADS / 32;  // 8
constexpr int32_t HASH_BITS       = 13;                    // 8192 entries
constexpr int32_t HASH_SIZE       = 1 << HASH_BITS;
constexpr int32_t HASH_MASK       = HASH_SIZE - 1;
constexpr int32_t MAX_BEAM        = 256;
constexpr int32_t MAX_DEGREE      = 128;
constexpr float   NEG_INF_CMNG    = -FLT_MAX;

// ─── Device helper: hash table operations ────────────────────────────
__device__ __forceinline__
bool hash_insert(int32_t* table, int32_t node) {
    // Open addressing with linear probing
    int32_t slot = node & HASH_MASK;
    for (int32_t probe = 0; probe < 32; ++probe) {
        int32_t old = atomicCAS(&table[slot], -1, node);
        if (old == -1 || old == node) return (old == -1);  // true = newly inserted
        slot = (slot + 1) & HASH_MASK;
    }
    return false;  // table full, treat as visited
}

__device__ __forceinline__
bool hash_contains(const int32_t* table, int32_t node) {
    int32_t slot = node & HASH_MASK;
    for (int32_t probe = 0; probe < 32; ++probe) {
        int32_t val = table[slot];
        if (val == node) return true;
        if (val == -1) return false;
        slot = (slot + 1) & HASH_MASK;
    }
    return false;
}

// ─── Device helper: warp-cooperative dot product ─────────────────────
// All 32 threads in a warp compute dot(a, b) for D-dimensional vectors.
__device__ __forceinline__
float warp_dot_product(const float* __restrict__ a,
                       const float* __restrict__ b,
                       int32_t D, int32_t lane) {
    float sum = 0.0f;
    for (int32_t d = lane; d < D; d += 32)
        sum += a[d] * b[d];
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    return sum;  // only lane 0 has the correct result
}

// FP16 variant
__device__ __forceinline__
float warp_dot_product_fp16(const half* __restrict__ a,
                            const float* __restrict__ b,
                            int32_t D, int32_t lane) {
    float sum = 0.0f;
    for (int32_t d = lane; d < D; d += 32)
        sum += __half2float(a[d]) * b[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    return sum;
}

// ─── Device helper: insert into sorted beam (descending by score) ────
// The beam is sorted descending: beam[0] = best, beam[beam_size-1] = worst.
// Only thread 0 should call this.
__device__ __forceinline__
bool beam_insert(float* beam_scores, int32_t* beam_ids,
                 int32_t beam_size, float score, int32_t node_id) {
    // Check if better than worst
    if (score <= beam_scores[beam_size - 1]) return false;

    // Replace worst
    beam_scores[beam_size - 1] = score;
    beam_ids[beam_size - 1] = node_id;

    // Bubble up to maintain descending sort
    for (int32_t j = beam_size - 2; j >= 0; --j) {
        if (beam_scores[j] < beam_scores[j + 1]) {
            float   ts = beam_scores[j]; beam_scores[j] = beam_scores[j+1]; beam_scores[j+1] = ts;
            int32_t ti = beam_ids[j];    beam_ids[j]    = beam_ids[j+1];    beam_ids[j+1]    = ti;
        } else break;
    }
    return true;
}

// ─── Main search kernel ──────────────────────────────────────────────
// Grid: 1 block per query. Block: SEARCH_THREADS threads.
// Shared memory: hash table + beam + query vector.
__global__ void cmng_beam_search_kernel(
    const int32_t* __restrict__ d_neighbors,     // N * degree
    const float*   __restrict__ d_embeddings,    // N * D
    const half*    __restrict__ d_embeddings_fp16, // N * D (may be null)
    const int32_t* __restrict__ d_modalities,    // N
    const float*   __restrict__ d_timestamps,    // N
    const float*   __restrict__ d_query,         // D
    int32_t*       __restrict__ d_result_ids,    // K (output)
    float*         __restrict__ d_result_scores, // K (output)
    int32_t*       __restrict__ d_result_mods,   // K (output)
    int32_t*       __restrict__ d_result_count,  // 1 (output)
    int32_t N, int32_t D, int32_t degree,
    int32_t beam_width, int32_t max_iterations, int32_t num_parents,
    int32_t K, float time_decay_lambda, float query_timestamp,
    bool use_fp16)
{
    // ── Shared memory layout ─────────────────────────────────────
    extern __shared__ char s_mem[];
    float*   s_beam_scores = (float*)s_mem;                              // beam_width
    int32_t* s_beam_ids    = (int32_t*)(s_beam_scores + beam_width);     // beam_width
    int32_t* s_beam_parent = (int32_t*)(s_beam_ids + beam_width);        // beam_width (0=unvisited parent)
    int32_t* s_hash        = (int32_t*)(s_beam_parent + beam_width);     // HASH_SIZE
    float*   s_query       = (float*)(s_hash + HASH_SIZE);               // D
    // Temp buffer for neighbor distances (degree floats)
    float*   s_neigh_dist  = s_query + D;                                // degree
    int32_t* s_neigh_ids   = (int32_t*)(s_neigh_dist + degree);          // degree

    int32_t tid  = threadIdx.x;
    int32_t warp = tid / 32;
    int32_t lane = tid & 31;

    // ── Load query into shared memory ────────────────────────────
    for (int32_t d = tid; d < D; d += SEARCH_THREADS)
        s_query[d] = d_query[d];

    // ── Initialize hash table ────────────────────────────────────
    for (int32_t i = tid; i < HASH_SIZE; i += SEARCH_THREADS)
        s_hash[i] = -1;

    // ── Initialize beam with random starting nodes ───────────────
    // Use a simple deterministic hash for reproducibility
    if (tid < beam_width) {
        s_beam_scores[tid] = NEG_INF_CMNG;
        s_beam_ids[tid] = -1;
        s_beam_parent[tid] = 0;
    }
    __syncthreads();

    // Seed: each warp picks a random starting node and computes distance
    {
        int32_t num_seeds = min(SEARCH_WARPS * num_parents, beam_width);
        if (warp < num_seeds) {
            // Deterministic seed based on query content
            uint32_t h = __float_as_uint(s_query[0]) ^ __float_as_uint(s_query[1]);
            int32_t seed_node = (int32_t)((h + warp * 7919u) % (uint32_t)N);

            float sim;
            if (use_fp16 && d_embeddings_fp16 != nullptr) {
                sim = warp_dot_product_fp16(
                    d_embeddings_fp16 + int64_t(seed_node) * D,
                    s_query, D, lane);
            } else {
                sim = warp_dot_product(
                    d_embeddings + int64_t(seed_node) * D,
                    s_query, D, lane);
            }

            if (lane == 0) {
                s_beam_scores[warp] = sim;
                s_beam_ids[warp] = seed_node;
                hash_insert(s_hash, seed_node);
            }
        }
    }
    __syncthreads();

    // ── Sort initial beam (thread 0, small bubble sort) ──────────
    if (tid == 0) {
        for (int32_t i = 0; i < beam_width - 1; ++i)
            for (int32_t j = i + 1; j < beam_width; ++j)
                if (s_beam_scores[j] > s_beam_scores[i]) {
                    float   ts = s_beam_scores[i]; s_beam_scores[i] = s_beam_scores[j]; s_beam_scores[j] = ts;
                    int32_t ti = s_beam_ids[i];    s_beam_ids[i]    = s_beam_ids[j];    s_beam_ids[j]    = ti;
                }
    }
    __syncthreads();

    // ── Main search loop ─────────────────────────────────────────
    for (int32_t iter = 0; iter < max_iterations; ++iter) {
        // Find the best unparented node in beam (thread 0)
        __shared__ int32_t s_parent_node;
        __shared__ int32_t s_changed;
        if (tid == 0) {
            s_parent_node = -1;
            s_changed = 0;
            for (int32_t i = 0; i < beam_width; ++i) {
                if (s_beam_ids[i] >= 0 && s_beam_parent[i] == 0) {
                    s_parent_node = s_beam_ids[i];
                    s_beam_parent[i] = 1;  // mark as parented
                    break;
                }
            }
        }
        __syncthreads();

        if (s_parent_node < 0) break;  // all candidates parented → converged

        // Load parent's neighbors into shared memory
        const int32_t* neigh_ptr = d_neighbors + int64_t(s_parent_node) * degree;
        for (int32_t i = tid; i < degree; i += SEARCH_THREADS) {
            s_neigh_ids[i] = neigh_ptr[i];
            s_neigh_dist[i] = NEG_INF_CMNG;
        }
        __syncthreads();

        // Each warp computes distance for one neighbor at a time
        // 8 warps process 8 neighbors in parallel, loop degree/8 times
        for (int32_t base = 0; base < degree; base += SEARCH_WARPS) {
            int32_t neigh_idx = base + warp;
            if (neigh_idx >= degree) continue;

            int32_t neigh_node = s_neigh_ids[neigh_idx];
            if (neigh_node < 0 || neigh_node >= N) continue;

            // Check if already visited (lane 0 checks hash)
            bool already_visited = false;
            if (lane == 0) {
                already_visited = hash_contains(s_hash, neigh_node);
            }
            already_visited = __shfl_sync(0xFFFFFFFF, already_visited ? 1 : 0, 0);
            if (already_visited) continue;

            // Compute distance
            float sim;
            if (use_fp16 && d_embeddings_fp16 != nullptr) {
                sim = warp_dot_product_fp16(
                    d_embeddings_fp16 + int64_t(neigh_node) * D,
                    s_query, D, lane);
            } else {
                sim = warp_dot_product(
                    d_embeddings + int64_t(neigh_node) * D,
                    s_query, D, lane);
            }

            if (lane == 0) {
                s_neigh_dist[neigh_idx] = sim;
                hash_insert(s_hash, neigh_node);
            }
        }
        __syncthreads();

        // Thread 0: insert new candidates into beam
        if (tid == 0) {
            for (int32_t i = 0; i < degree; ++i) {
                if (s_neigh_ids[i] < 0 || s_neigh_dist[i] <= NEG_INF_CMNG * 0.5f) continue;
                bool inserted = beam_insert(s_beam_scores, s_beam_ids,
                                            beam_width, s_neigh_dist[i], s_neigh_ids[i]);
                if (inserted) s_changed = 1;
            }
        }
        __syncthreads();

        // If nothing changed, we might be converged — but keep going
        // until we exhaust unparented nodes (the break at top handles this)
    }

    // ── Apply temporal decay to beam ─────────────────────────────
    if (tid < beam_width && s_beam_ids[tid] >= 0) {
        int32_t node = s_beam_ids[tid];
        float age = fmaxf(0.0f, query_timestamp - d_timestamps[node]);
        s_beam_scores[tid] *= __expf(-time_decay_lambda * age);
    }
    __syncthreads();

    // ── Re-sort by decayed score (thread 0) ──────────────────────
    if (tid == 0) {
        // Simple selection sort for top-K (K is small, typically 10)
        for (int32_t i = 0; i < min(K, beam_width) - 1; ++i) {
            int32_t best = i;
            for (int32_t j = i + 1; j < beam_width; ++j)
                if (s_beam_scores[j] > s_beam_scores[best]) best = j;
            if (best != i) {
                float   ts = s_beam_scores[i]; s_beam_scores[i] = s_beam_scores[best]; s_beam_scores[best] = ts;
                int32_t ti = s_beam_ids[i];    s_beam_ids[i]    = s_beam_ids[best];    s_beam_ids[best]    = ti;
            }
        }

        // Write top-K results
        int32_t count = 0;
        for (int32_t i = 0; i < min(K, beam_width); ++i) {
            if (s_beam_ids[i] < 0) break;
            d_result_ids[i] = s_beam_ids[i];
            d_result_scores[i] = s_beam_scores[i];
            d_result_mods[i] = d_modalities[s_beam_ids[i]];
            count++;
        }
        *d_result_count = count;
    }
}

// ─── Host-side search context ────────────────────────────────────────
CMNGSearchContext create_cmng_context(int32_t D, int32_t max_beam, int32_t max_k) {
    CMNGSearchContext ctx;
    ctx.max_D = D;
    ctx.max_K = max_k;
    ctx.max_beam = max_beam;

    CMNG_CHECK(cudaMalloc(&ctx.d_result_ids,    max_k * sizeof(int32_t)));
    CMNG_CHECK(cudaMalloc(&ctx.d_result_scores, max_k * sizeof(float)));
    CMNG_CHECK(cudaMalloc(&ctx.d_result_mods,   max_k * sizeof(int32_t)));
    CMNG_CHECK(cudaMalloc(&ctx.d_result_count,  sizeof(int32_t)));
    CMNG_CHECK(cudaMalloc(&ctx.d_query,         D * sizeof(float)));
    CMNG_CHECK(cudaMallocHost(&ctx.h_query_pinned, D * sizeof(float)));

    cudaEventCreate(&ctx.e_start);
    cudaEventCreate(&ctx.e_end);
    cudaStreamCreate(&ctx.stream);

    ctx.valid = true;
    return ctx;
}

void destroy_cmng_context(CMNGSearchContext& ctx) {
    if (!ctx.valid) return;
    cudaFree(ctx.d_result_ids);
    cudaFree(ctx.d_result_scores);
    cudaFree(ctx.d_result_mods);
    cudaFree(ctx.d_result_count);
    cudaFree(ctx.d_query);
    cudaFreeHost(ctx.h_query_pinned);
    cudaEventDestroy(ctx.e_start);
    cudaEventDestroy(ctx.e_end);
    cudaStreamDestroy(ctx.stream);
    ctx = {};
}

// ─── Host-side search driver ─────────────────────────────────────────
std::vector<QueryResult>
cmng_search(const CMNGGraph&      g,
            CMNGSearchContext&     ctx,
            const float*          h_query,
            float                 query_timestamp,
            const CMNGSearchConfig& cfg,
            RetrievalStats&       stats)
{
    int32_t K = std::min(cfg.K, ctx.max_K);
    int32_t beam = std::min(cfg.beam_width, (int32_t)MAX_BEAM);
    cudaStream_t s = ctx.stream;

    // H2D query copy (pinned → async)
    std::memcpy(ctx.h_query_pinned, h_query, g.D * sizeof(float));
    CMNG_CHECK(cudaMemcpyAsync(ctx.d_query, ctx.h_query_pinned,
                                g.D * sizeof(float), cudaMemcpyHostToDevice, s));

    // Compute shared memory size
    int32_t smem = beam * sizeof(float)       // s_beam_scores
                 + beam * sizeof(int32_t)     // s_beam_ids
                 + beam * sizeof(int32_t)     // s_beam_parent
                 + HASH_SIZE * sizeof(int32_t) // s_hash
                 + g.D * sizeof(float)         // s_query
                 + g.degree * sizeof(float)    // s_neigh_dist
                 + g.degree * sizeof(int32_t); // s_neigh_ids

    cudaEventRecord(ctx.e_start, s);

    cmng_beam_search_kernel<<<1, SEARCH_THREADS, smem, s>>>(
        g.d_neighbors, g.d_embeddings, g.d_embeddings_fp16,
        g.d_modalities, g.d_timestamps, ctx.d_query,
        ctx.d_result_ids, ctx.d_result_scores, ctx.d_result_mods,
        ctx.d_result_count,
        g.N, g.D, g.degree,
        beam, cfg.max_iterations, cfg.num_parents,
        K, cfg.time_decay_lambda, query_timestamp,
        cfg.use_fp16);
    CMNG_CHECK(cudaGetLastError());

    cudaEventRecord(ctx.e_end, s);
    CMNG_CHECK(cudaStreamSynchronize(s));

    // Read results
    int32_t count = 0;
    CMNG_CHECK(cudaMemcpy(&count, ctx.d_result_count, sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
    count = std::min(count, K);

    std::vector<int32_t> h_ids(count);
    std::vector<float>   h_scores(count);
    std::vector<int32_t> h_mods(count);
    if (count > 0) {
        CMNG_CHECK(cudaMemcpy(h_ids.data(), ctx.d_result_ids,
                               count * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CMNG_CHECK(cudaMemcpy(h_scores.data(), ctx.d_result_scores,
                               count * sizeof(float), cudaMemcpyDeviceToHost));
        CMNG_CHECK(cudaMemcpy(h_mods.data(), ctx.d_result_mods,
                               count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    std::vector<QueryResult> results;
    results.reserve(count);
    for (int32_t i = 0; i < count; ++i)
        results.push_back({ h_ids[i], h_scores[i], h_mods[i], 0 });

    // Timing
    cudaEventElapsedTime(&stats.gpu_ms_total, ctx.e_start, ctx.e_end);
    stats.gpu_ms_similarity = stats.gpu_ms_total;  // single kernel, no breakdown
    stats.gpu_ms_topk = 0;
    stats.gpu_ms_bfs = 0;
    stats.gpu_ms_rerank = 0;
    stats.nodes_scanned = 0;  // not tracked in beam search
    stats.bfs_waves = 0;

    return results;
}
