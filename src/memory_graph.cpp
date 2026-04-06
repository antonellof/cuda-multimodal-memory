// ============================================================================
//  memory_graph.cpp
// ============================================================================
#include "memory_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_set>

MemoryGraph MemoryGraph::synthetic(int32_t n_text,
                                   int32_t n_audio,
                                   int32_t n_image,
                                   int32_t dim,
                                   uint32_t seed) {
    MemoryGraph g;
    g.num_nodes     = n_text + n_audio + n_image;
    g.embedding_dim = dim;
    g.embeddings.resize(static_cast<size_t>(g.num_nodes) * dim);
    g.modalities.resize(g.num_nodes);
    g.timestamps.resize(g.num_nodes);

    std::mt19937 rng(seed);
    std::normal_distribution<float>       N01(0.0f, 1.0f);
    std::uniform_real_distribution<float> T(0.0f, 1e7f);

    // Generate embeddings clustered by modality: each modality has its own
    // mean vector; embeddings are drawn N(mean_mod, I) then L2-normalized.
    // This mimics how CLIP / CLAP / text encoders produce modality-local
    // clusters in a shared latent space.
    std::vector<std::vector<float>> mod_means(MOD_COUNT, std::vector<float>(dim));
    for (int m = 0; m < MOD_COUNT; ++m)
        for (int d = 0; d < dim; ++d)
            mod_means[m][d] = N01(rng) * 0.5f;

    auto fill_nodes = [&](int32_t start, int32_t count, int32_t modality) {
        for (int32_t i = 0; i < count; ++i) {
            int32_t idx = start + i;
            g.modalities[idx] = modality;
            g.timestamps[idx] = T(rng);
            float norm = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float v = mod_means[modality][d] + N01(rng);
                g.embeddings[size_t(idx) * dim + d] = v;
                norm += v * v;
            }
            norm = std::sqrt(norm) + 1e-8f;
            for (int d = 0; d < dim; ++d)
                g.embeddings[size_t(idx) * dim + d] /= norm;
        }
    };
    int32_t off = 0;
    fill_nodes(off, n_text,  MOD_TEXT);  off += n_text;
    fill_nodes(off, n_audio, MOD_AUDIO); off += n_audio;
    fill_nodes(off, n_image, MOD_IMAGE);
    return g;
}

// ─── NSN edges with configurable phase enables ──────────────────────
void MemoryGraph::build_nsn_edges(int32_t k, double p) {
    NSNConfig cfg;
    cfg.k = k;
    cfg.p = p;
    build_nsn_edges_configurable(cfg);
}

void MemoryGraph::build_nsn_edges_configurable(const NSNConfig& cfg) {
    const int32_t N = num_nodes;
    const int32_t k = cfg.k;
    const double  p = cfg.p;
    std::vector<std::unordered_set<int32_t>> s(N);

    auto add = [&](int32_t a, int32_t b) {
        if (a == b) return;
        s[a].insert(b);
        s[b].insert(a);
    };

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_int_distribution<int32_t> RN(0, N - 1);

    // Phase 1: ring lattice
    if (cfg.enable_ring) {
        for (int32_t i = 0; i < N; ++i)
            for (int32_t j = 1; j <= k / 2; ++j)
                add(i, (i + j) % N);
    }

    // Phase 2: hierarchical skip connections
    if (cfg.enable_skips) {
        int32_t levels = std::max(1, int32_t(std::log2(double(N))));
        for (int32_t lvl = 1; lvl <= levels; ++lvl) {
            int32_t step = 1 << lvl;
            for (int32_t i = 0; i < N; i += step)
                add(i, (i + step) % N);
        }
    }

    // Phase 3: hub supernodes at sqrt(N) intervals
    if (cfg.enable_hubs) {
        int32_t hub_int = std::max(1, int32_t(std::sqrt(double(N))));
        int32_t extras  = std::max(2, int32_t(std::log2(double(N))) / 2);
        for (int32_t h = 0; h < N; h += hub_int)
            for (int32_t e = 0; e < extras; ++e)
                add(h, RN(rng));
    }

    // Phase 4: small-world rewiring
    if (cfg.enable_rewire && cfg.enable_ring) {
        for (int32_t i = 0; i < N; ++i) {
            for (int32_t j = 1; j <= k / 2; ++j) {
                if (U(rng) < p) {
                    int32_t old_t = (i + j) % N;
                    if (s[i].count(old_t) && s[i].size() > 2) {
                        s[i].erase(old_t); s[old_t].erase(i);
                        add(i, RN(rng));
                    }
                }
            }
        }
    }

    // Phase 5: MULTIMODAL BRIDGES — for every memory, add one extra edge to
    // a randomly chosen memory of each OTHER modality. This guarantees that
    // a text query can reach an audio or image memory in bounded hops along
    // the cross-modal backbone.
    if (cfg.enable_bridges) {
        std::vector<std::vector<int32_t>> by_mod(MOD_COUNT);
        for (int32_t i = 0; i < N; ++i) by_mod[modalities[i]].push_back(i);

        for (int32_t i = 0; i < N; ++i) {
            int32_t my_mod = modalities[i];
            for (int32_t m = 0; m < MOD_COUNT; ++m) {
                if (m == my_mod || by_mod[m].empty()) continue;
                int32_t target = by_mod[m][rng() % by_mod[m].size()];
                add(i, target);
            }
        }
    }

    // ─── Convert adjacency sets to CSR ───
    row_offsets.assign(N + 1, 0);
    for (int32_t i = 0; i < N; ++i)
        row_offsets[i + 1] = row_offsets[i] + int32_t(s[i].size());
    num_edges = row_offsets[N];
    col_indices.resize(num_edges);
    for (int32_t i = 0; i < N; ++i) {
        std::vector<int32_t> row(s[i].begin(), s[i].end());
        std::sort(row.begin(), row.end());
        std::copy(row.begin(), row.end(),
                  col_indices.begin() + row_offsets[i]);
    }
}

void MemoryGraph::print_summary() const {
    int32_t counts[MOD_COUNT] = {0};
    for (int32_t m : modalities) counts[m]++;
    std::printf("  MemoryGraph: N=%d  E=%d  D=%d  avg_deg=%.2f\n",
                num_nodes, num_edges / 2, embedding_dim,
                num_nodes ? double(num_edges) / num_nodes : 0.0);
    std::printf("    Text=%d  Audio=%d  Image=%d\n",
                counts[MOD_TEXT], counts[MOD_AUDIO], counts[MOD_IMAGE]);
    std::printf("    Device memory: %.2f MB\n", device_bytes() / (1024.0 * 1024.0));
}

size_t MemoryGraph::device_bytes() const {
    return row_offsets.size() * sizeof(int32_t)
         + col_indices.size() * sizeof(int32_t)
         + embeddings.size()  * sizeof(float)
         + modalities.size()  * sizeof(int32_t)
         + timestamps.size()  * sizeof(float);
}
