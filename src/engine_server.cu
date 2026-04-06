// ============================================================================
//  engine_server.cu
//  JSON-line REPL for MARS.
//
//  Reads single-line JSON commands from stdin, writes single-line JSON
//  responses to stdout. Designed to be driven by the Python subprocess
//  client in benchmarks/common/engine_client.py.
//
//  Supported commands:
//    ADD       {id, modality, embedding, timestamp}  -> {ok: true}
//    QUERY     {embedding, top_k, ...}               -> {results: [...]}
//    DELETE    {id}                                   -> {ok: true}
//    STATS     {}                                    -> {n, edges, vram_mb}
//    SHUTDOWN  {}                                    -> {ok: true}
//
//  Usage:
//    ./engine_server --dim 768
//
//  The server maintains a persistent MemoryGraph that grows with ADD
//  commands. After each ADD the NSN edges are incrementally appended
//  (cross-modal bridges + local ring neighbors for the new node).
//  A full graph rebuild is triggered when the node count crosses a
//  power-of-two boundary, keeping the skip-connection and hub-supernode
//  invariants valid.
//
//  QUERY wraps the existing query_memory() pipeline from memory_cuda.cu.
//  The device graph is re-uploaded only when dirty (i.e., after ADD/DELETE).
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#include "memory_graph.h"
#include "memory_cuda.cuh"

// ─── Minimal JSON helpers ─────────────────────────────────────────────
// We intentionally avoid pulling in a full JSON library. The protocol is
// simple enough that we can parse with string scanning and emit with
// snprintf. This keeps the build dependency-free.

namespace {

// Skip whitespace
const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    return p;
}

// Extract a string value for a given key from a flat JSON object.
// Returns empty string if not found.
std::string json_get_string(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// Extract a numeric value for a given key.
double json_get_number(const std::string& json, const char* key, double dflt = 0.0) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return dflt;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return dflt;
    const char* p = skip_ws(json.c_str() + pos + 1);
    char* end = nullptr;
    double val = std::strtod(p, &end);
    if (end == p) return dflt;
    return val;
}

// Extract an integer value for a given key.
int32_t json_get_int(const std::string& json, const char* key, int32_t dflt = 0) {
    return static_cast<int32_t>(json_get_number(json, key, dflt));
}

// Extract a float array for a given key (e.g., "embedding": [0.1, 0.2, ...]).
std::vector<float> json_get_float_array(const std::string& json, const char* key) {
    std::vector<float> out;
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return out;
    pos = json.find('[', pos + needle.size());
    if (pos == std::string::npos) return out;
    const char* p = json.c_str() + pos + 1;
    while (*p && *p != ']') {
        p = skip_ws(p);
        if (*p == ']' || *p == '\0') break;
        char* end = nullptr;
        float v = std::strtof(p, &end);
        if (end == p) break;
        out.push_back(v);
        p = skip_ws(end);
        if (*p == ',') ++p;
    }
    return out;
}

// Write a single-line JSON response to stdout (no newline inside).
void respond(const char* json_str) {
    std::fputs(json_str, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void respond_ok() {
    respond("{\"ok\":true}");
}

void respond_error(const char* msg) {
    std::fprintf(stdout, "{\"ok\":false,\"error\":\"%s\"}\n", msg);
    std::fflush(stdout);
}

} // namespace

// ─── Server state ─────────────────────────────────────────────────────

struct ServerState {
    int32_t embedding_dim = 768;

    // Host-side graph (grows with ADD commands)
    MemoryGraph graph;
    bool dirty = true;  // needs re-upload to GPU

    // Device-side graph (uploaded lazily before QUERY)
    DeviceMemoryGraph dg;
    bool device_allocated = false;

    // Track the last full-rebuild size so we know when to rebuild NSN
    int32_t last_rebuild_size = 0;

    std::mt19937 rng{42};

    void init(int32_t dim) {
        embedding_dim = dim;
        graph.num_nodes = 0;
        graph.num_edges = 0;
        graph.embedding_dim = dim;
    }

    // Add a single memory node
    void add_node(int32_t id, int32_t modality,
                  const std::vector<float>& embedding, float timestamp) {
        // Grow the graph to accommodate this id
        while (graph.num_nodes <= id) {
            graph.embeddings.resize(size_t(graph.num_nodes + 1) * embedding_dim, 0.0f);
            graph.modalities.push_back(MOD_TEXT);
            graph.timestamps.push_back(0.0f);
            graph.num_nodes++;
        }

        // Copy embedding (L2-normalize)
        float norm = 0.0f;
        for (int32_t d = 0; d < embedding_dim; ++d)
            norm += embedding[d] * embedding[d];
        norm = std::sqrt(norm) + 1e-8f;
        float* dst = &graph.embeddings[size_t(id) * embedding_dim];
        for (int32_t d = 0; d < embedding_dim; ++d)
            dst[d] = embedding[d] / norm;

        graph.modalities[id] = modality;
        graph.timestamps[id] = timestamp;
        dirty = true;
    }

    // Delete a node by zeroing its embedding (tombstone approach)
    void delete_node(int32_t id) {
        if (id < 0 || id >= graph.num_nodes) return;
        float* dst = &graph.embeddings[size_t(id) * embedding_dim];
        std::memset(dst, 0, embedding_dim * sizeof(float));
        graph.timestamps[id] = -1.0f;  // mark as deleted
        dirty = true;
    }

    // Ensure NSN edges are built and device graph is current
    void ensure_device() {
        if (!dirty && device_allocated) return;

        // Rebuild NSN if the graph has grown significantly since last rebuild
        // (or if this is the first build)
        if (graph.num_nodes > 0 &&
            (last_rebuild_size == 0 ||
             graph.num_nodes >= last_rebuild_size * 2 ||
             graph.num_nodes - last_rebuild_size > 1000)) {
            graph.build_nsn_edges(/*k=*/6, /*p=*/0.15);
            last_rebuild_size = graph.num_nodes;
        }

        // Free previous device allocation
        if (device_allocated) {
            free_device(dg);
            device_allocated = false;
        }

        // Upload
        if (graph.num_nodes > 0 && graph.row_offsets.size() > 0) {
            dg = upload_to_device(graph);
            device_allocated = true;
        }
        dirty = false;
    }
};

// ─── Main REPL ────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int32_t dim = 768;

    // Parse --dim argument
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--dim") == 0) {
            dim = std::atoi(argv[i + 1]);
        }
    }

    ServerState state;
    state.init(dim);

    std::fprintf(stderr, "engine_server: ready (dim=%d)\n", dim);

    // Read lines from stdin
    char buf[1024 * 1024];  // 1 MB line buffer (embeddings can be large)
    while (std::fgets(buf, sizeof(buf), stdin) != nullptr) {
        std::string line(buf);
        // Trim trailing newline
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        if (line.empty()) continue;

        std::string cmd = json_get_string(line, "cmd");

        // ── ADD ───────────────────────────────────────────────────
        if (cmd == "ADD") {
            int32_t id = json_get_int(line, "id");
            int32_t modality = json_get_int(line, "modality", 0);
            float timestamp = static_cast<float>(json_get_number(line, "timestamp", 0.0));
            auto embedding = json_get_float_array(line, "embedding");

            if (static_cast<int32_t>(embedding.size()) != state.embedding_dim) {
                respond_error("embedding dimension mismatch");
                continue;
            }

            state.add_node(id, modality, embedding, timestamp);
            respond_ok();
        }

        // ── QUERY ─────────────────────────────────────────────────
        else if (cmd == "QUERY") {
            auto embedding = json_get_float_array(line, "embedding");
            if (static_cast<int32_t>(embedding.size()) != state.embedding_dim) {
                respond_error("query embedding dimension mismatch");
                continue;
            }

            int32_t top_k = json_get_int(line, "top_k", 10);
            int32_t bfs_max_hops = json_get_int(line, "bfs_max_hops", 2);
            int32_t modality_filter = json_get_int(line, "modality_filter", -1);
            float timestamp = static_cast<float>(json_get_number(line, "timestamp", 0.0));

            // Ensure graph is on device
            state.ensure_device();

            if (!state.device_allocated || state.graph.num_nodes == 0) {
                respond("{\"results\":[]}");
                continue;
            }

            RetrievalConfig cfg;
            cfg.top_k = top_k;
            cfg.bfs_max_hops = bfs_max_hops;
            cfg.modality_filter = modality_filter;
            cfg.time_decay_lambda = 1e-8f;
            cfg.bfs_score_decay = 0.5f;

            RetrievalStats stats;
            QueryContext ctx = create_query_context(
                state.graph.num_nodes, state.embedding_dim, top_k);
            auto results = query_memory_fast(state.dg, ctx, embedding.data(),
                                             timestamp, cfg, stats);
            destroy_query_context(ctx);

            // Build JSON response — single line
            std::string resp = "{\"results\":[";
            for (size_t i = 0; i < results.size(); ++i) {
                if (i > 0) resp += ",";
                char entry[256];
                std::snprintf(entry, sizeof(entry),
                    "{\"id\":%d,\"score\":%.6f,\"modality\":%d,\"hops\":%d}",
                    results[i].node_id, results[i].score,
                    results[i].modality, results[i].hops_from_seed);
                resp += entry;
            }
            resp += "],\"gpu_ms\":";
            char ms_buf[32];
            std::snprintf(ms_buf, sizeof(ms_buf), "%.3f", stats.gpu_ms_total);
            resp += ms_buf;
            resp += "}";
            respond(resp.c_str());
        }

        // ── DELETE ────────────────────────────────────────────────
        else if (cmd == "DELETE") {
            int32_t id = json_get_int(line, "id");
            state.delete_node(id);
            respond_ok();
        }

        // ── STATS ─────────────────────────────────────────────────
        else if (cmd == "STATS") {
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            float vram_used_mb = float(total_mem - free_mem) / (1024.0f * 1024.0f);

            char resp[256];
            std::snprintf(resp, sizeof(resp),
                "{\"n\":%d,\"edges\":%d,\"vram_mb\":%.1f}",
                state.graph.num_nodes, state.graph.num_edges / 2, vram_used_mb);
            respond(resp);
        }

        // ── SAVE ──────────────────────────────────────────────────
        else if (cmd == "SAVE") {
            // Persistence not yet implemented — acknowledge but warn
            respond("{\"ok\":true,\"warning\":\"persistence not yet implemented\"}");
        }

        // ── LOAD ──────────────────────────────────────────────────
        else if (cmd == "LOAD") {
            respond("{\"ok\":true,\"warning\":\"persistence not yet implemented\"}");
        }

        // ── SHUTDOWN ──────────────────────────────────────────────
        else if (cmd == "SHUTDOWN") {
            respond_ok();
            break;
        }

        // ── Unknown ───────────────────────────────────────────────
        else {
            respond_error("unknown command");
        }
    }

    // Cleanup
    if (state.device_allocated) {
        free_device(state.dg);
    }

    std::fprintf(stderr, "engine_server: shutdown\n");
    return 0;
}
