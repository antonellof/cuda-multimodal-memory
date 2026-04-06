// ============================================================================
//  cmng_build.cu — Cross-Modal Navigable Graph construction
//
//  Phase 1: Tiled brute-force k-NN via cuBLAS SGEMM
//  Phase 2: Select top-(degree - degree_cross) + insert cross-modal bridges
//
//  Construction is offline / one-time. The goal is correctness and
//  simplicity for the MVP; NN-descent for N > 200K is a follow-up.
// ============================================================================
#include "cmng.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ─── Constants ───────────────────────────────────────────────────────
constexpr int32_t BUILD_THREADS = 256;
constexpr int32_t TILE_SIZE     = 4096;   // tile size for SGEMM k-NN

// ─── Kernel: merge SGEMM tile results into running k-NN list ────────
// Each block handles one row (one query node) of the tile.
// d_knn_ids[node * k_init .. (node+1)*k_init - 1] = running best IDs
// d_knn_dists[...] = running best distances (sorted descending: worst first)
// d_sim_tile[row * tile_cols + col] = similarity for (node, candidate)
__global__ void knn_merge_tile_kernel(
    float*         __restrict__ d_knn_dists,  // N * k_init
    int32_t*       __restrict__ d_knn_ids,    // N * k_init
    const float*   __restrict__ d_sim_tile,   // tile_rows * tile_cols
    int32_t                      tile_row_start,
    int32_t                      tile_col_start,
    int32_t                      tile_rows,
    int32_t                      tile_cols,
    int32_t                      k_init,
    int32_t                      N)
{
    int32_t local_row = blockIdx.x;
    if (local_row >= tile_rows) return;
    int32_t node = tile_row_start + local_row;
    if (node >= N) return;

    float*   knn_d = d_knn_dists + int64_t(node) * k_init;
    int32_t* knn_i = d_knn_ids   + int64_t(node) * k_init;

    // Each thread scans a stripe of the tile columns
    for (int32_t c = threadIdx.x; c < tile_cols; c += blockDim.x) {
        int32_t candidate = tile_col_start + c;
        if (candidate >= N || candidate == node) continue;

        float sim = d_sim_tile[int64_t(local_row) * tile_cols + c];

        // Find the worst (minimum similarity) in the k-NN list
        // knn_d is sorted descending, so the last element is the worst
        float worst = knn_d[k_init - 1];
        if (sim > worst) {
            // Replace worst entry (simple, correct, not optimal)
            knn_d[k_init - 1] = sim;
            knn_i[k_init - 1] = candidate;

            // Bubble up to maintain descending sort
            for (int32_t j = k_init - 2; j >= 0; --j) {
                if (knn_d[j] < knn_d[j + 1]) {
                    float   td = knn_d[j]; knn_d[j] = knn_d[j+1]; knn_d[j+1] = td;
                    int32_t ti = knn_i[j]; knn_i[j] = knn_i[j+1]; knn_i[j+1] = ti;
                } else break;
            }
        }
    }
}

// ─── Kernel: select final neighbors + cross-modal bridges ───────────
// Each block handles one node.
__global__ void select_neighbors_kernel(
    const int32_t* __restrict__ d_knn_ids,    // N * k_init (sorted by sim desc)
    const int32_t* __restrict__ d_modalities, // N
    int32_t*       __restrict__ d_neighbors,  // N * degree (output)
    int32_t                      N,
    int32_t                      k_init,
    int32_t                      degree,
    int32_t                      degree_cross,
    int32_t                      num_modalities)
{
    int32_t node = blockIdx.x;
    if (node >= N) return;

    const int32_t* knn = d_knn_ids + int64_t(node) * k_init;
    int32_t* out = d_neighbors + int64_t(node) * degree;
    int32_t my_mod = d_modalities[node];

    int32_t intra_slots = degree - degree_cross;

    // Thread 0 does the selection (simple for correctness)
    if (threadIdx.x == 0) {
        // Fill intra-modal slots: top-(degree - degree_cross) nearest
        int32_t filled = 0;
        for (int32_t i = 0; i < k_init && filled < intra_slots; ++i) {
            int32_t cand = knn[i];
            if (cand < 0 || cand >= N) continue;
            out[filled++] = cand;
        }
        // Pad remaining intra slots with -1
        for (int32_t i = filled; i < intra_slots; ++i)
            out[i] = -1;

        // Fill cross-modal bridge slots: closest node per other modality
        // We need degree_cross / (num_modalities - 1) per modality,
        // but for simplicity, one per modality then fill remainder
        int32_t bridge_idx = intra_slots;
        for (int32_t m = 0; m < num_modalities && bridge_idx < degree; ++m) {
            if (m == my_mod) continue;
            // Find closest node of modality m in k-NN list
            bool found = false;
            for (int32_t i = 0; i < k_init; ++i) {
                int32_t cand = knn[i];
                if (cand < 0 || cand >= N) continue;
                if (d_modalities[cand] == m) {
                    out[bridge_idx++] = cand;
                    found = true;
                    break;
                }
            }
            if (!found && bridge_idx < degree) {
                out[bridge_idx++] = -1;  // no node of this modality found
            }
        }
        // Pad remaining bridge slots
        for (int32_t i = bridge_idx; i < degree; ++i)
            out[i] = -1;
    }
}

// ─── Host function: build_cmng ───────────────────────────────────────
CMNGGraph build_cmng(const float*   h_embeddings,
                     const int32_t* h_modalities,
                     const float*   h_timestamps,
                     int32_t N, int32_t D,
                     int32_t degree,
                     int32_t degree_cross)
{
    CMNGGraph g;
    g.N = N;
    g.D = D;
    g.degree = degree;
    g.degree_cross = degree_cross;

    int32_t k_init = degree * 2;  // initial k-NN degree (2x final)

    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);

    // ── Upload embeddings + metadata ─────────────────────────────
    CMNG_CHECK(cudaMalloc(&g.d_embeddings,  int64_t(N) * D * sizeof(float)));
    CMNG_CHECK(cudaMalloc(&g.d_modalities,  N * sizeof(int32_t)));
    CMNG_CHECK(cudaMalloc(&g.d_timestamps,  N * sizeof(float)));
    CMNG_CHECK(cudaMalloc(&g.d_neighbors,   int64_t(N) * degree * sizeof(int32_t)));

    CMNG_CHECK(cudaMemcpy(g.d_embeddings, h_embeddings,
                           int64_t(N) * D * sizeof(float), cudaMemcpyHostToDevice));
    CMNG_CHECK(cudaMemcpy(g.d_modalities, h_modalities,
                           N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CMNG_CHECK(cudaMemcpy(g.d_timestamps, h_timestamps,
                           N * sizeof(float), cudaMemcpyHostToDevice));

    // ── Phase 1: Tiled brute-force k-NN via cuBLAS SGEMM ────────
    // Allocate k-NN storage (sorted descending by similarity)
    float*   d_knn_dists;
    int32_t* d_knn_ids;
    CMNG_CHECK(cudaMalloc(&d_knn_dists, int64_t(N) * k_init * sizeof(float)));
    CMNG_CHECK(cudaMalloc(&d_knn_ids,   int64_t(N) * k_init * sizeof(int32_t)));

    // Initialize k-NN with -infinity distances and -1 IDs
    {
        std::vector<float>   h_init_d(int64_t(N) * k_init, -1e30f);
        std::vector<int32_t> h_init_i(int64_t(N) * k_init, -1);
        CMNG_CHECK(cudaMemcpy(d_knn_dists, h_init_d.data(),
                               int64_t(N) * k_init * sizeof(float), cudaMemcpyHostToDevice));
        CMNG_CHECK(cudaMemcpy(d_knn_ids, h_init_i.data(),
                               int64_t(N) * k_init * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    // Allocate SGEMM output tile
    int32_t max_tile = std::min(N, TILE_SIZE);
    float* d_sim_tile;
    CMNG_CHECK(cudaMalloc(&d_sim_tile, int64_t(max_tile) * max_tile * sizeof(float)));

    // cuBLAS handle
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Tiled all-pairs: for each (tile_i, tile_j), compute sim = E_i * E_j^T
    int32_t num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    float alpha = 1.0f, beta = 0.0f;

    std::fprintf(stderr, "  CMNG build: %d nodes, %d tiles, k_init=%d, degree=%d\n",
                 N, num_tiles * num_tiles, k_init, degree);

    for (int32_t ti = 0; ti < num_tiles; ++ti) {
        int32_t row_start = ti * TILE_SIZE;
        int32_t row_count = std::min(TILE_SIZE, N - row_start);

        for (int32_t tj = 0; tj < num_tiles; ++tj) {
            int32_t col_start = tj * TILE_SIZE;
            int32_t col_count = std::min(TILE_SIZE, N - col_start);

            // SGEMM: sim_tile[row_count x col_count] = E_rows[row_count x D] * E_cols[D x col_count]
            // E is row-major: E_rows starts at g.d_embeddings + row_start * D
            // cuBLAS expects column-major, so we compute:
            // C = B^T * A  where A = E_rows^T (D x row_count), B = E_cols^T (D x col_count)
            // C^T = A^T * B = E_rows * E_cols^T  (row_count x col_count, row-major)
            // In cuBLAS col-major: C(col_count x row_count) = E_cols(col_count x D) * E_rows^T(D x row_count)
            cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        col_count, row_count, D,
                        &alpha,
                        g.d_embeddings + int64_t(col_start) * D, D,
                        g.d_embeddings + int64_t(row_start) * D, D,
                        &beta,
                        d_sim_tile, col_count);

            // Merge tile results into running k-NN
            knn_merge_tile_kernel<<<row_count, BUILD_THREADS>>>(
                d_knn_dists, d_knn_ids, d_sim_tile,
                row_start, col_start, row_count, col_count,
                k_init, N);
            CMNG_CHECK(cudaGetLastError());
        }
    }
    CMNG_CHECK(cudaDeviceSynchronize());

    // ── Phase 2: Select final neighbors + cross-modal bridges ────
    int32_t num_modalities = 3;  // MOD_TEXT, MOD_AUDIO, MOD_IMAGE
    select_neighbors_kernel<<<N, BUILD_THREADS>>>(
        d_knn_ids, g.d_modalities, g.d_neighbors,
        N, k_init, degree, degree_cross, num_modalities);
    CMNG_CHECK(cudaGetLastError());
    CMNG_CHECK(cudaDeviceSynchronize());

    // ── Cleanup construction temporaries ─────────────────────────
    cudaFree(d_knn_dists);
    cudaFree(d_knn_ids);
    cudaFree(d_sim_tile);
    cublasDestroy(cublas);

    cudaEventRecord(t_end);
    cudaEventSynchronize(t_end);
    cudaEventElapsedTime(&g.build_time_ms, t_start, t_end);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);

    std::fprintf(stderr, "  CMNG build complete: %.1f ms\n", g.build_time_ms);
    return g;
}

// ─── FP16 upload ─────────────────────────────────────────────────────
extern __global__ void convert_f32_to_f16_kernel(const float* __restrict__ in,
                                                  half* __restrict__ out,
                                                  int32_t count);

void cmng_upload_fp16(CMNGGraph& g) {
    int64_t count = int64_t(g.N) * g.D;
    if (count <= 0 || g.d_embeddings == nullptr) return;
    CMNG_CHECK(cudaMalloc(&g.d_embeddings_fp16, count * sizeof(half)));
    int32_t blocks = (int32_t)((count + 255) / 256);
    convert_f32_to_f16_kernel<<<blocks, 256>>>(
        g.d_embeddings, g.d_embeddings_fp16, (int32_t)count);
    CMNG_CHECK(cudaGetLastError());
    CMNG_CHECK(cudaDeviceSynchronize());
}

// ─── Free ────────────────────────────────────────────────────────────
void free_cmng(CMNGGraph& g) {
    if (g.d_neighbors)       cudaFree(g.d_neighbors);
    if (g.d_embeddings)      cudaFree(g.d_embeddings);
    if (g.d_embeddings_fp16) cudaFree(g.d_embeddings_fp16);
    if (g.d_modalities)      cudaFree(g.d_modalities);
    if (g.d_timestamps)      cudaFree(g.d_timestamps);
    g = {};
}
