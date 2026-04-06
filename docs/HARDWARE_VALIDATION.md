# Hardware Validation Guide

This project has been validated on **four GPUs**:

| GPU | VRAM | Architecture | CUDA | Status |
|-----|------|-------------|------|--------|
| A100X 80GB | 80 GB | Ampere sm_80 | 12.0 | validated |
| A100 PCIE 40GB | 40 GB | Ampere sm_80 | 12.8 | validated |
| A100 SXM4 80GB | 80 GB | Ampere sm_80 | 12.6–12.8 | validated |
| RTX 5060 Ti 16GB | 16 GB | Blackwell sm_120 | 13.0 | validated |

All raw results are in the [`results/`](../results/) folder.

This document explains how to reproduce the validation on rented GPU
hardware — no local GPU required.

**Total cost: under $3. Total time: under 30 minutes.**

---

## What the validation harness does

Running `make check` builds and runs `src/validate.cu`, which:

1. Reports the GPU device properties (name, compute capability, SM count,
   HBM bandwidth, driver version)
2. Runs the full four-stage retrieval pipeline across 5 corpus sizes
   (1K to 16K memories) with 768-D embeddings
3. Measures per-stage latency with `cudaEvent` timing (sub-microsecond accuracy)
4. Performs a correctness check: compares the GPU's top-10 against a
   brute-force CPU top-10 and reports recall@10
5. Writes a structured `results/results.json` to the current directory

The output JSON is self-contained and ready to paste into the paper or
attach to a GitHub issue.

## Full benchmark suite

Beyond `make check`, the repo includes several benchmark tiers:

```bash
# Core validation (5 corpus sizes, 100 queries each)
make check                   # -> results/results.json

# Demo-level GPU kernel timing (4 workloads)
make demo-av                 # 60 Hz, 2.4K memories, 1 ms budget
make demo-robot              # 1 kHz, 10K memories, 1 ms budget
make demo-ar                 # 90 Hz, 27K memories, 5 ms budget
make demo-voice              # 30 Hz, 9K memories, 20 ms budget

# Wall-clock latency benchmarks (includes kernel launch overhead)
make bench-av                # 10s, 2.4K memories
make bench-robot             # 10s, 6K memories
make bench-ar                # 50s, 20K memories
make bench-voice             # 30s, 3K memories

# Sustained duration tests (real-world length)
make bench-sustained         # all four 15-30s tests

# Corpus-size scaling sweep
make bench-scale             # 60 Hz × 15s at N = 1K, 5K, 10K, 20K, 50K

# GPU keepalive variants (for low-rate workloads)
make bench-av-keepalive      # 10s, 2.4K memories, keepalive ON
make bench-av-30s-keepalive  # 30s, 5K memories, keepalive ON
```

All benchmark outputs go to the `results/` folder as JSON.

---

## Renting a GPU instance

### Option 1 — vast.ai (cheapest)

**Typical cost:** $0.30–$0.80/hour for RTX 4090, $0.80–$1.50/hour for A100

1. Sign up at [vast.ai](https://vast.ai/) and add $5 of credit
2. **Console > Instances**, search for:
   - **GPU:** `RTX 4090`, `A100 PCIE`, `A100 SXM4`, or `H100`
   - **Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` (any CUDA 12+ devel image)
   - **Disk:** 10 GB is plenty
3. Click **Rent** and wait for boot
4. Connect via SSH

```bash
apt-get update && apt-get install -y git build-essential
git clone https://github.com/antonellof/MARS.git
cd MARS
make info                    # verify nvcc + GPU
make                         # build everything
make check                   # core validation -> results/results.json

# Full suite (optional, ~5 min total)
make demo-av && make demo-robot && make demo-ar && make demo-voice
make bench-av && make bench-robot && make bench-ar && make bench-voice
make bench-sustained
make bench-scale
```

5. **Destroy the instance** immediately after — vast.ai bills by the second
6. Download results before destroying:

```bash
# From your local machine
scp -P <port> root@<host>:/root/MARS/results/*.json .
```

### Option 2 — Lambda Labs

**Typical cost:** $1.10/hour for A100 PCIE 40GB, $1.89/hour for A100 SXM4 80GB

```bash
pip install lambda-cloud
lambda-cloud gpu launch --instance-type gpu_1x_a100_pcie --region us-west-1
ssh ubuntu@<instance-ip>

git clone https://github.com/antonellof/MARS.git
cd MARS && make && make check

scp ubuntu@<instance-ip>:~/MARS/results/*.json .
lambda-cloud gpu terminate <instance-id>
```

### Option 3 — RunPod

**Typical cost:** $0.44/hour for RTX A5000, $1.89/hour for A100 80GB

1. Sign up at [runpod.io](https://runpod.io/)
2. Deploy a **Pod** with:
   - **GPU:** `RTX A6000`, `A100 80GB`, or `H100 PCIe`
   - **Template:** `RunPod PyTorch 2.1` (includes CUDA 12)
   - **Disk:** 10 GB
3. Click **Connect > Web Terminal**
4. Same build and validation steps as above
5. Download results via the web UI or `runpodctl send`
6. **Stop the pod** when done

### Option 4 — Paperspace Gradient

**Typical cost:** $1.15/hour for A100, $2.30/hour for H100

1. Sign up at [paperspace.com](https://paperspace.com/)
2. **Gradient > Notebooks > Create Notebook**
3. Choose an A100 machine type and the PyTorch template
4. Open a terminal inside the notebook
5. Same build and validation steps
6. Download results via the notebook file browser

---

## Reference results (A100X, 80 GB)

These are the measured results from the project's A100X validation run.
Your results should be in the same ballpark (within 2x for different GPUs).

| N      | Similarity | Rerank | Top-K  | BFS    | Total  | QPS   | Recall@10 |
|--------|-----------|--------|--------|--------|--------|-------|-----------|
| 1,000  | 0.014 ms  | 0.010  | 0.564  | 0.095  | 0.684  | 1,463 | 1.00      |
| 2,000  | 0.016 ms  | 0.009  | 0.505  | 0.097  | 0.627  | 1,596 | 0.90      |
| 4,000  | 0.017 ms  | 0.008  | 0.412  | 0.085  | 0.522  | 1,917 | 0.80      |
| 8,000  | 0.028 ms  | 0.008  | 0.803  | 0.088  | 0.927  | 1,079 | 0.80      |
| 16,000 | 0.061 ms  | 0.008  | 0.803  | 0.085  | 0.957  | 1,045 | 0.60      |

---

## Submitting your results

After a successful run, please open a GitHub issue with:

1. The full contents of `results/results.json`
2. The output of `make info` (GPU name, nvcc version)
3. The output of `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`
4. Any unexpected warnings from the build or runtime

Your results will be added to the README's benchmark table with attribution.

---

## Troubleshooting

### `nvcc: command not found`
Use a `*-devel` image (`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`), not
a `*-runtime` image. Alternatively: `apt-get install -y nvidia-cuda-toolkit`

### `nvidia-smi: command not found`
The container was not launched with GPU access. On vast.ai this should be
automatic. On Docker, you need `--gpus all`.

### `error: unsupported gpu architecture 'compute_70'`
CUDA toolkit too old. Need 11.8+. Check with `nvcc --version`.

### Build succeeds but `./validate` crashes immediately
Likely a compute-capability mismatch. The Makefile targets sm_70–sm_89.
For H100 (sm_90), add `-gencode arch=compute_90,code=sm_90` to the
`GENCODE` list in the Makefile.

### `recall_at_k` is below 0.9
Correctness issue on your GPU. Most common cause is an `atomicCAS`
pointer-type mismatch on older toolchains. Please open a GitHub issue.

### The build produces a warning flood
The `--expt-relaxed-constexpr` flag suppresses most warnings. If you still
see thousands of lines, your nvcc is probably pre-11.8.

---

## Advanced: custom parameters

```bash
./validate <n_queries_per_config> <embedding_dim>
```

Defaults: 100 queries, 768 dimensions. Examples:

```bash
./validate 500 1024    # 500 queries, 1024-D embeddings
./validate 50  512     # 50 queries, 512-D embeddings
```
