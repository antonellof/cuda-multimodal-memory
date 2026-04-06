// ============================================================================
//  demos/common/frame_timer.h
//
//  Shared infrastructure for MARS demos
//  demonstrators. Each demo uses this header to:
//    - time per-frame retrieval latency
//    - collect a histogram
//    - compute p50 / p90 / p99 / max
//    - assert a hard deadline (p99 < budget)
//    - emit a JSON report to stdout
//
//  Also includes a synthetic sensor embedding generator that produces
//  clustered embeddings simulating object identity tracks across frames.
// ============================================================================
#ifndef DEMO_FRAME_TIMER_H
#define DEMO_FRAME_TIMER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace demo {

// ─── Latency histogram ──────────────────────────────────────────────
struct LatencyHistogram {
    std::vector<double> samples_ms;

    void add(double ms)                { samples_ms.push_back(ms); }
    size_t count() const               { return samples_ms.size(); }

    double percentile(double p) const {
        if (samples_ms.empty()) return 0.0;
        std::vector<double> sorted = samples_ms;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
        return sorted[idx];
    }

    double p50() const  { return percentile(0.50); }
    double p90() const  { return percentile(0.90); }
    double p99() const  { return percentile(0.99); }

    double max_ms() const {
        return samples_ms.empty() ? 0.0
             : *std::max_element(samples_ms.begin(), samples_ms.end());
    }

    double mean_ms() const {
        if (samples_ms.empty()) return 0.0;
        double s = 0.0;
        for (double v : samples_ms) s += v;
        return s / samples_ms.size();
    }
};

// ─── Demo report ────────────────────────────────────────────────────
struct DemoReport {
    std::string demo_name;
    std::string scenario;
    int32_t     frame_rate_hz    = 0;
    double      deadline_p99_ms  = 0.0;
    int32_t     num_frames       = 0;
    int32_t     corpus_size      = 0;
    int32_t     embedding_dim    = 0;

    LatencyHistogram latencies;

    // Workload-specific metrics (key -> value), e.g. "track_recovery_rate"
    std::vector<std::pair<std::string, double>> metrics;

    void add_metric(const std::string& key, double value) {
        metrics.emplace_back(key, value);
    }

    bool passed() const {
        return latencies.count() > 0 && latencies.p99() <= deadline_p99_ms;
    }

    void print_json() const {
        std::printf("{\n");
        std::printf("  \"demo\": \"%s\",\n", demo_name.c_str());
        std::printf("  \"scenario\": \"%s\",\n", scenario.c_str());
        std::printf("  \"frame_rate_hz\": %d,\n", frame_rate_hz);
        std::printf("  \"deadline_p99_ms\": %.3f,\n", deadline_p99_ms);
        std::printf("  \"num_frames\": %d,\n", num_frames);
        std::printf("  \"corpus_size\": %d,\n", corpus_size);
        std::printf("  \"embedding_dim\": %d,\n", embedding_dim);
        std::printf("  \"latency_ms\": {\n");
        std::printf("    \"p50\":  %.4f,\n", latencies.p50());
        std::printf("    \"p90\":  %.4f,\n", latencies.p90());
        std::printf("    \"p99\":  %.4f,\n", latencies.p99());
        std::printf("    \"max\":  %.4f,\n", latencies.max_ms());
        std::printf("    \"mean\": %.4f\n",  latencies.mean_ms());
        std::printf("  },\n");
        std::printf("  \"metrics\": {\n");
        for (size_t i = 0; i < metrics.size(); ++i) {
            std::printf("    \"%s\": %.4f%s\n",
                        metrics[i].first.c_str(),
                        metrics[i].second,
                        (i + 1 < metrics.size()) ? "," : "");
        }
        std::printf("  },\n");
        std::printf("  \"deadline_met\": %s\n", passed() ? "true" : "false");
        std::printf("}\n");
    }

    void print_summary() const {
        std::fprintf(stderr, "\n═══════════════════════════════════════════════════════════\n");
        std::fprintf(stderr, "  %s\n", demo_name.c_str());
        std::fprintf(stderr, "  %s\n", scenario.c_str());
        std::fprintf(stderr, "───────────────────────────────────────────────────────────\n");
        std::fprintf(stderr, "  Frame rate:       %d Hz\n", frame_rate_hz);
        std::fprintf(stderr, "  Frames processed: %d\n", num_frames);
        std::fprintf(stderr, "  Corpus size:      %d memories\n", corpus_size);
        std::fprintf(stderr, "  Latency p50:      %.3f ms\n", latencies.p50());
        std::fprintf(stderr, "  Latency p90:      %.3f ms\n", latencies.p90());
        std::fprintf(stderr, "  Latency p99:      %.3f ms\n", latencies.p99());
        std::fprintf(stderr, "  Latency max:      %.3f ms\n", latencies.max_ms());
        std::fprintf(stderr, "  Deadline (p99):   %.3f ms\n", deadline_p99_ms);
        for (auto& kv : metrics) {
            std::fprintf(stderr, "  %-18s%.3f\n", (kv.first + ":").c_str(), kv.second);
        }
        std::fprintf(stderr, "───────────────────────────────────────────────────────────\n");
        if (passed()) {
            std::fprintf(stderr, "  \x1b[32mDEADLINE MET ✓\x1b[0m  (p99 %.3f ms < budget %.3f ms)\n",
                         latencies.p99(), deadline_p99_ms);
        } else {
            std::fprintf(stderr, "  \x1b[31mDEADLINE MISSED ✗\x1b[0m  (p99 %.3f ms > budget %.3f ms)\n",
                         latencies.p99(), deadline_p99_ms);
        }
        std::fprintf(stderr, "═══════════════════════════════════════════════════════════\n\n");
    }
};

// ─── Synthetic sensor embedding generator ───────────────────────────
// Produces embedding vectors that cluster by object identity, with
// frame-to-frame drift simulating continuous tracking.
class SyntheticSensor {
public:
    SyntheticSensor(int32_t dim, int32_t num_identities, uint32_t seed = 42)
        : dim_(dim), rng_(seed), N_(0.0f, 1.0f) {
        identities_.resize(num_identities);
        for (auto& id : identities_) {
            id.resize(dim);
            for (int i = 0; i < dim; ++i) id[i] = N_(rng_) * 0.5f;
            normalize(id.data());
        }
    }

    // Sample an embedding for a specific identity with optional drift
    std::vector<float> sample(int32_t identity_id, float drift = 0.1f) {
        std::vector<float> out(dim_);
        const std::vector<float>& base = identities_[identity_id % identities_.size()];
        for (int i = 0; i < dim_; ++i) out[i] = base[i] + N_(rng_) * drift;
        normalize(out.data());
        return out;
    }

    // Random unrelated embedding (background noise)
    std::vector<float> sample_random() {
        std::vector<float> out(dim_);
        for (int i = 0; i < dim_; ++i) out[i] = N_(rng_);
        normalize(out.data());
        return out;
    }

    int32_t dim() const { return dim_; }

private:
    void normalize(float* v) {
        float norm = 0.0f;
        for (int i = 0; i < dim_; ++i) norm += v[i] * v[i];
        norm = std::sqrt(norm) + 1e-8f;
        for (int i = 0; i < dim_; ++i) v[i] /= norm;
    }

    int32_t dim_;
    std::mt19937 rng_;
    std::normal_distribution<float> N_;
    std::vector<std::vector<float>> identities_;
};

} // namespace demo

#endif // DEMO_FRAME_TIMER_H
