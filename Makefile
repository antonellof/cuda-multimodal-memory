# ============================================================================
#  Makefile — MARS (Memory for Autonomous Real-time Systems)
#
#  A GPU-resident multimodal memory substrate for real-time AI systems.
#
#  Targets:
#    make              Build everything (engine, validate, latency_bench, demos)
#    make engine       Build the interactive benchmark CLI
#    make validate     Build the JSON-emitting validation harness
#    make latency      Build the deadline-aware latency benchmark
#    make server       Build the JSON-line REPL engine server
#    make demos        Build all four application demonstrators
#    make tests        Build and run host-only unit tests (no GPU required)
#    make check        Run the validation harness and save to results.json
#    make run          Run the interactive CLI at a default size
#    make clean        Remove all build artifacts
#    make info         Print nvcc version and GPU info (for vast.ai verify)
#
#  Demo-specific targets (each runs at its natural sensor rate):
#    make demo-av        ./demos/av_perception/demo    (60 Hz,  1 ms target)
#    make demo-robot     ./demos/robot_episodic/demo   (1 kHz,  1 ms target)
#    make demo-ar        ./demos/ar_spatial/demo       (90 Hz,  5 ms target)
#    make demo-voice     ./demos/voice_agent/demo      (30 Hz, 20 ms target)
#
#  Latency benchmark presets (match the four application demonstrators):
#    make bench-av       ./latency_bench   60 1.0   600   2400
#    make bench-robot    ./latency_bench 1000 1.0 10000   6000
#    make bench-ar       ./latency_bench   90 5.0  4500  20000
#    make bench-voice    ./latency_bench   30 20.0  900   3000
#
#  Sustained long-running benchmarks (real-world duration):
#    make bench-sustained   Run all four 30-second sustained tests
#    make bench-av-30s      AV perception, 30s real-time, 5K memories
#    make bench-robot-15s   Humanoid control, 15s real-time, 10K memories
#    make bench-ar-30s      AR/VR headset, 30s real-time, 50K memories
#    make bench-voice-30s   Voice agent, 30s real-time, 10K memories
#
#  Scaling stress test (sweep corpus sizes):
#    make bench-scale       60 Hz × 15s at N = 1K, 5K, 10K, 20K, 50K
#
#  Requirements:
#    CUDA Toolkit 11.8+, nvcc in PATH
#    NVIDIA GPU with compute capability 7.0+
#    C++17 host compiler (gcc 9+, clang 10+)
#
#  For tests target only: just g++, no CUDA, no GPU.
# ============================================================================

NVCC      ?= nvcc
CXX       ?= g++

CXXFLAGS  := -std=c++17 -O3 -Iinclude
DEMO_INC  := -Iinclude -Idemos/common
GENCODE   := -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89
NVCCFLAGS := $(CXXFLAGS) $(GENCODE) --expt-relaxed-constexpr

# ─── Source files ────────────────────────────────────────────────────
COMMON_OBJS    := src/memory_graph.o src/memory_cuda.o
ENGINE_OBJS    := $(COMMON_OBJS) src/main.o
VALIDATE_OBJS  := $(COMMON_OBJS) src/validate.o
CMNG_OBJS      := src/cmng_build.o src/cmng_search.o
LATENCY_OBJS   := $(COMMON_OBJS) $(CMNG_OBJS) src/latency_bench.o
SERVER_OBJS    := $(COMMON_OBJS) src/engine_server.o

DEMO_NAMES  := av_perception robot_episodic ar_spatial voice_agent
DEMO_BINS   := $(foreach d,$(DEMO_NAMES),demos/$(d)/demo)

# ─── Primary targets ─────────────────────────────────────────────────
.PHONY: all engine validate latency server demos tests run check clean info paper
.PHONY: demo-av demo-robot demo-ar demo-voice
.PHONY: bench-av bench-robot bench-ar bench-voice
.PHONY: bench-sustained bench-av-30s bench-robot-15s bench-ar-30s bench-voice-30s
.PHONY: bench-scale bench-large bench-fp16 bench-cmng bench-mars bench-ablation bench-ablation-quick
.PHONY: bench-av-keepalive bench-av-30s-keepalive

all: engine validate latency server demos

engine:   memory_engine
validate: validate
latency:  latency_bench
server:   engine_server
demos:    $(DEMO_BINS)

# ─── Binary rules ────────────────────────────────────────────────────
memory_engine: $(ENGINE_OBJS)
	$(NVCC) $(NVCCFLAGS) -lcublas -o $@ $^

validate: $(VALIDATE_OBJS)
	$(NVCC) $(NVCCFLAGS) -lcublas -o validate $^

latency_bench: $(LATENCY_OBJS)
	$(NVCC) $(NVCCFLAGS) -lcublas -o latency_bench $^

engine_server: $(SERVER_OBJS)
	$(NVCC) $(NVCCFLAGS) -lcublas -o engine_server $^

# Each demo builds from its demo.cu + common objects + frame_timer.h
demos/%/demo: demos/%/demo.cu $(COMMON_OBJS) demos/common/frame_timer.h include/memory_cuda.cuh include/memory_graph.h
	$(NVCC) $(NVCCFLAGS) $(DEMO_INC) -lcublas -o $@ $< $(COMMON_OBJS)

# ─── Object file rules ───────────────────────────────────────────────
src/%.o: src/%.cpp include/memory_graph.h
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

src/%.o: src/%.cu include/memory_graph.h include/memory_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

src/cmng_build.o: src/cmng_build.cu include/cmng.cuh include/memory_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

src/cmng_search.o: src/cmng_search.cu include/cmng.cuh include/memory_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ─── Tests (host-only, NO CUDA NEEDED) ───────────────────────────────
tests:
	$(CXX) -std=c++17 -Iinclude -O2 -o tests/run_tests \
	    src/memory_graph.cpp tests/test_memory_graph.cpp
	./tests/run_tests

# ─── Convenience run targets ─────────────────────────────────────────
run: memory_engine
	./memory_engine 4000 2000 2000 768

check: validate
	@mkdir -p results
	./validate 100 768 | tee results/results.json
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Validation complete. Results written to results/results.json"
	@echo "═══════════════════════════════════════════════════════════"

# Demo runs (each at its natural sensor rate + deadline target)
demo-av:    demos/av_perception/demo
	./demos/av_perception/demo

demo-robot: demos/robot_episodic/demo
	./demos/robot_episodic/demo

demo-ar:    demos/ar_spatial/demo
	./demos/ar_spatial/demo

demo-voice: demos/voice_agent/demo
	./demos/voice_agent/demo

# Latency benchmark presets (rate, budget_ms, frames, corpus)
bench-av:    latency_bench
	@mkdir -p results
	./latency_bench   60  1.0   600  2400   | tee results/bench_av.json

bench-robot: latency_bench
	@mkdir -p results
	./latency_bench 1000  1.0 10000  6000   | tee results/bench_robot.json

bench-ar:    latency_bench
	@mkdir -p results
	./latency_bench   90  5.0  4500 20000   | tee results/bench_ar.json

bench-voice: latency_bench
	@mkdir -p results
	./latency_bench   30 20.0   900  3000   | tee results/bench_voice.json

# Keepalive variants (prevent GPU clock-state drops at low sensor rates)
bench-av-keepalive: latency_bench
	@mkdir -p results
	./latency_bench   60  1.0   600  2400 --keepalive | tee results/bench_av_keepalive.json

bench-av-30s-keepalive: latency_bench
	@mkdir -p results
	./latency_bench   60  1.0  1800  5000 --keepalive | tee results/sustained_av_30s_keepalive.json

# ─── Paper (requires tectonic: brew install tectonic) ────────────
paper:
	cd paper && tectonic --reruns 3 main.tex
	@echo "  ✓ paper/main.pdf rebuilt"

# ─── Diagnostics ─────────────────────────────────────────────────────
info:
	@echo "NVCC:     $(NVCC)"
	@$(NVCC) --version 2>/dev/null | tail -1 || echo "  (nvcc not found)"
	@echo "CXX:      $(CXX)"
	@$(CXX)  --version 2>/dev/null | head -1 || echo "  (g++ not found)"
	@echo "GPU(s):"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"

# ─── Sustained long-running benchmarks ────────────────────────────────
# Real-world durations: 15-30 seconds of wall-clock time at sensor rate.
# These stress-test thermal throttling, memory allocator pressure, and
# OS scheduling interference that short runs may not expose.

bench-av-30s: latency_bench
	@mkdir -p results
	@echo "══════════ AV Perception — 30s sustained ══════════"
	./latency_bench   60  1.0  1800   5000  | tee results/sustained_av_30s.json

bench-robot-15s: latency_bench
	@mkdir -p results
	@echo "══════════ Robot Episodic — 15s sustained ══════════"
	./latency_bench 1000  1.0 15000  10000  | tee results/sustained_robot_15s.json

bench-ar-30s: latency_bench
	@mkdir -p results
	@echo "══════════ AR/VR Spatial — 30s sustained ══════════"
	./latency_bench   90  5.0  2700  50000  | tee results/sustained_ar_30s.json

bench-voice-30s: latency_bench
	@mkdir -p results
	@echo "══════════ Voice Agent — 30s sustained ══════════"
	./latency_bench   30 20.0   900  10000  | tee results/sustained_voice_30s.json

bench-sustained: bench-av-30s bench-robot-15s bench-ar-30s bench-voice-30s
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  All sustained benchmarks complete."
	@echo "  Results in results/ folder"
	@echo "═══════════════════════════════════════════════════"

# ─── Scaling stress test ─────────────────────────────────────────────
# Sweep corpus sizes at a fixed 60 Hz rate for 15 seconds (900 frames).
# Shows how latency scales with working-set size.

bench-scale: latency_bench
	@mkdir -p results
	@echo "══════════ Scaling: N=1K ══════════"
	./latency_bench 60 1.0  900   1000  | tee results/scale_1k.json
	@echo "══════════ Scaling: N=5K ══════════"
	./latency_bench 60 1.0  900   5000  | tee results/scale_5k.json
	@echo "══════════ Scaling: N=10K ══════════"
	./latency_bench 60 1.0  900  10000  | tee results/scale_10k.json
	@echo "══════════ Scaling: N=20K ══════════"
	./latency_bench 60 1.0  900  20000  | tee results/scale_20k.json
	@echo "══════════ Scaling: N=50K ══════════"
	./latency_bench 60 1.0  900  50000  | tee results/scale_50k.json
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  Scaling sweep complete."
	@echo "  Results in results/ folder"
	@echo "═══════════════════════════════════════════════════"

# ─── Large corpus scaling (v4 FP16 + CUDA graph) ────────────────────
# Tests how the system scales from 50K to 2M memories with FP16 enabled.
# Use --fp16 --graph for maximum performance at large N.

bench-large: latency_bench
	@mkdir -p results
	@echo "══════════ Large: N=50K (FP16+Graph) ══════════"
	./latency_bench 60 5.0  300  50000   --fp16 --graph | tee results/large_50k.json
	@echo "══════════ Large: N=100K (FP16+Graph) ══════════"
	./latency_bench 60 5.0  300  100000  --fp16 --graph | tee results/large_100k.json
	@echo "══════════ Large: N=200K (FP16+Graph) ══════════"
	./latency_bench 60 10.0 300  200000  --fp16 --graph | tee results/large_200k.json
	@echo "══════════ Large: N=500K (FP16+Graph) ══════════"
	./latency_bench 60 20.0 300  500000  --fp16 --graph | tee results/large_500k.json
	@echo "══════════ Large: N=1M (FP16+Graph) ══════════"
	./latency_bench 60 50.0 100  1000000 --fp16 --graph | tee results/large_1m.json
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  Large corpus sweep complete."
	@echo "═══════════════════════════════════════════════════"

# ─── FP16 vs FP32 A/B comparison ───────────────────────────────────
# Same workload (60 Hz, 10K, 15s) with and without FP16.

bench-fp16: latency_bench
	@mkdir -p results
	@echo "══════════ FP32 baseline (N=10K) ══════════"
	./latency_bench 60 1.0  900  10000 | tee results/fp32_10k.json
	@echo "══════════ FP16 (N=10K) ══════════"
	./latency_bench 60 1.0  900  10000 --fp16 | tee results/fp16_10k.json
	@echo "══════════ FP16+Graph (N=10K) ══════════"
	./latency_bench 60 1.0  900  10000 --fp16 --graph | tee results/fp16_graph_10k.json
	@echo "══════════ FP32 baseline (N=50K) ══════════"
	./latency_bench 60 5.0  900  50000 | tee results/fp32_50k.json
	@echo "══════════ FP16 (N=50K) ══════════"
	./latency_bench 60 5.0  900  50000 --fp16 | tee results/fp16_50k.json
	@echo "══════════ FP16+Graph (N=50K) ══════════"
	./latency_bench 60 5.0  900  50000 --fp16 --graph | tee results/fp16_graph_50k.json
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  FP16 A/B comparison complete."
	@echo "═══════════════════════════════════════════════════"

# ─── CMNG graph ANN benchmarks (v5) ──────────────────────────────────
# Greedy beam search on cross-modal navigable graph.
# Compares CMNG vs brute-force at various corpus sizes.

bench-cmng: latency_bench
	@mkdir -p results
	@echo "══════════ CMNG N=10K ══════════"
	./latency_bench 60 1.0 300  10000  --cmng | tee results/cmng_10k.json
	@echo "══════════ CMNG N=50K ══════════"
	./latency_bench 60 5.0 300  50000  --cmng | tee results/cmng_50k.json
	@echo "══════════ CMNG N=100K ══════════"
	./latency_bench 60 5.0 300  100000 --cmng | tee results/cmng_100k.json
	@echo "══════════ CMNG N=200K ══════════"
	./latency_bench 60 10.0 300 200000 --cmng | tee results/cmng_200k.json
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  CMNG benchmark sweep complete."
	@echo "═══════════════════════════════════════════════════"

# ─── MARS benchmark: cuBLAS similarity + CUB radix-sort top-K ────────
# Targets FAISS-competitive latency via elimination of the top-K bottleneck.
# Sweeps N = 2.4K, 10K, 20K, 50K to show scaling.

bench-mars: latency_bench
	@mkdir -p results
	@echo "══════════ MARS N=2.4K (AV perception) ══════════"
	./latency_bench 60 1.0  600   2400  --mars | tee results/mars_2400.json
	@echo "══════════ MARS N=10K ══════════"
	./latency_bench 60 1.0  600  10000  --mars | tee results/mars_10k.json
	@echo "══════════ MARS N=20K ══════════"
	./latency_bench 60 5.0  600  20000  --mars | tee results/mars_20k.json
	@echo "══════════ MARS N=50K ══════════"
	./latency_bench 60 5.0  300  50000  --mars | tee results/mars_50k.json
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  MARS benchmark sweep complete."
	@echo "  Results in results/mars_*.json"
	@echo "═══════════════════════════════════════════════════"

# ─── Ablation study (NSN topology contribution) ─────────────────────
# Tests 6 hypotheses about NSN design choices.
# Each variant runs at 60 Hz × 300 frames with --recall to measure
# recall@10 and cross-modal hit rate alongside latency.
#
# Variants:
#   full         — all 5 NSN phases, h=2 (baseline)
#   no_bridges   — phases 1-4 only (no cross-modal edges)
#   no_hubs      — phases 1,2,4,5 (no hub supernodes)
#   flat         — no graph, brute-force top-K only (h=0)
#   h0           — full NSN but BFS disabled (h=0)
#   h1           — full NSN, 1-hop BFS
#   h3           — full NSN, 3-hop BFS
#
# Output: results/ablation_*.json (one file per variant × corpus size)

.PHONY: bench-ablation

bench-ablation: latency_bench
	@mkdir -p results/ablation
	@echo "══════════ NSN Ablation Study ══════════"
	@for N in 2000 5000 10000 50000; do \
		for variant in full no_bridges no_hubs flat h0 h1 h3; do \
			echo "── $$variant N=$$N ──"; \
			./latency_bench 60 99.0 300 $$N --recall --ablate $$variant \
				| tee results/ablation/$${variant}_$${N}.json; \
		done; \
	done
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  Ablation study complete. Results in results/ablation/"
	@echo "  Parse with: python3 scripts/parse_ablation.py"
	@echo "═══════════════════════════════════════════════════"

# Quick ablation at small N (for testing, ~2 min)
bench-ablation-quick: latency_bench
	@mkdir -p results/ablation
	@for variant in full no_bridges no_hubs flat h0 h1 h3; do \
		echo "── $$variant N=5000 ──"; \
		./latency_bench 60 99.0 200 5000 --recall --ablate $$variant \
			| tee results/ablation/$${variant}_5000.json; \
	done

clean:
	rm -f src/*.o
	rm -f memory_engine validate latency_bench engine_server
	rm -f $(DEMO_BINS)
	rm -f tests/run_tests
	rm -rf results/
