# GP-Directed PSO with ML Particle Repositioning

GPU-accelerated Particle Swarm Optimization, guided by a Gaussian Process surrogate and an ML-trained classifier that periodically detects and relocates unproductive ("stuck") particles. Built to scale to 50+ dimensions on consumer GPUs and supercomputers.

Implements and extends the six GP-Directed PSO variants from [arXiv:2102.04172](https://arxiv.org/abs/2102.04172), adding a trained LightGBM classifier that fires every K iterations to rebalance exploration and exploitation.

---

## What's in it

| Component | Role |
|---|---|
| `Swarm` (PyTorch) | Tensorized particles — one batched tensor per state. |
| `GPSurrogate` (GPyTorch) | Exact GP with ARD-RBF kernel; batched autograd acquisition searches. |
| `StuckClassifier` (LightGBM) | Trained offline on ~500 PSO trajectories across dimensions {5, 10, 20, 50} and ten landscapes. Detects stuck / unproductive particles from 11 features. |
| `MLRepositioner` | Every K iterations, flags the top 20 % of particles by P(stuck) and teleports them — half to the GP's LCB minimum (exploit), half to the GP's max-uncertainty point (explore). |
| Six variants (A1–C2) | From the paper. All composable with `--use-ml`. |

---

## Requirements

- NVIDIA GPU with CUDA 12.x driver (consumer cards like RTX 4070 work; A100/H100 on clusters work)
- Linux (native) or WSL2 on Windows
- 8 GB RAM, 5 GB disk for training artifacts
- Conda / Mamba

macOS or CPU-only machines can run the code (it falls back transparently), but training-data generation and benchmark runs will be much slower.

---

## Installation

### Linux (native)

```bash
git clone https://github.com/<your-user>/<repo>.git
cd <repo>
conda env create -f environment.yml
conda activate pso-gpu
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True, <your GPU>
```

### Windows + NVIDIA GPU (RTX 4070, etc.) — via WSL2

Native Windows is **not supported** because LightGBM's GPU build is broken outside Linux. Use WSL2.

**One-time setup (Windows host, PowerShell as administrator):**

```powershell
# 1. Make sure your NVIDIA Windows driver is R525+ (Feb 2023 or newer).
# 2. Install WSL2 with Ubuntu 22.04:
wsl --install -d Ubuntu-22.04
# Reboot if prompted, then launch Ubuntu from the Start menu.
```

**Inside the WSL2 Ubuntu terminal:**

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate

# Clone the repo INSIDE WSL (not on /mnt/c — file I/O is ~10× slower there)
git clone https://github.com/<your-user>/<repo>.git ~/pso
cd ~/pso

# Create env + verify GPU passthrough
conda env create -f environment.yml
conda activate pso-gpu
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True, NVIDIA GeForce RTX 4070
```

**Common WSL2 gotchas:**
- Do **not** install CUDA inside WSL. The Windows driver forwards the GPU automatically; installing CUDA in WSL causes conflicts.
- Keep the repo under `~/` (the Linux filesystem), not `/mnt/c/...`.
- If `torch.cuda.is_available()` returns `False`, update your Windows NVIDIA driver and run `wsl --update` in PowerShell.

### Supercomputer (Linux + SLURM)

```bash
# Log in to the cluster
ssh you@cluster.example.edu

# Clone into your scratch space
cd $SCRATCH
git clone https://github.com/<your-user>/<repo>.git
cd <repo>

# Load modules (edit slurm/*.sbatch to match your cluster's module system)
module load anaconda/2024.02   # or whatever's available
module load cuda/12.1

conda env create -f environment.yml
conda activate pso-gpu
```

Then edit the three `slurm/*.sbatch` files: uncomment the `module load` lines, adjust `--time`, `--cpus-per-task`, partition/account flags as needed by your cluster.

### macOS / CPU-only

```bash
git clone https://github.com/<your-user>/<repo>.git
cd <repo>
conda env create -f environment.yml
conda activate pso-gpu
# All commands below work with --device cpu but are significantly slower
```

---

## Quick start

One command handles everything (data generation, training, benchmark):

```bash
bash scripts/run_local.sh 10 51 cuda:0      # 10 dimensions, 51 seeds, GPU 0
```

That script:
1. Generates `training_data.npz` if missing (~1–2 h on RTX 4070).
2. Trains `stuck_classifier.lgb` if missing (~10–30 min).
3. Runs the full benchmark and writes `out/results_dim10.jsonl`.
4. Prints the aggregated summary.

Cached artifacts (`.npz` and `.lgb`) are reused on subsequent invocations — no re-training per run.

---

## Manual pipeline

If you prefer step-by-step:

```bash
# 1. Generate training data (one-time)
python generate_training_data.py --runs 500 --device cuda:0 --out training_data.npz

# 2. Train classifier (one-time)
python train_classifier.py --data training_data.npz --out stuck_classifier.lgb

# 3. Smoke test (single run)
python main_gp.py --dim 10 --variant A3 --use-ml --device cuda:0

# 4. Benchmark (51 seeds × 6 variants × 5 functions × ML on/off)
python experiment.py --dim 10 --seeds 51 --device cuda:0 \
    --out out/results_dim10.jsonl

# 5. Print summary
python aggregate_results.py out/results_dim10.jsonl
```

---

## Supercomputer workflow

```bash
sbatch slurm/generate_data.sbatch                 # ~4 h on one GPU
sbatch --dependency=afterok:<prev_job_id> slurm/train.sbatch

# Array job: one GPU per seed, all 51 run in parallel
sbatch --export=DIM=50,ALL slurm/experiment.sbatch

python aggregate_results.py out/*.jsonl
```

---

## Smoke tests

Each core module has built-in assertions. Run these first to verify your install:

```bash
python function.py        # function values + numpy cross-check
python swarm.py           # PSO on 2D sphere converges to < 1e-3
python gp_surrogate.py    # GP fits a quadratic, argmin near origin
```

All three should print `checks passed` and exit 0.

---

## CLI reference

Most scripts share these flags:

| Flag | Default | Description |
|---|---|---|
| `--dim` | 10 | Problem dimensionality. Budget = 200 × dim evaluations. |
| `--device` | `cuda:0` | PyTorch device. Falls back to CPU if CUDA unavailable. |
| `--seed` | 42 | RNG seed (single-run scripts). |
| `--seeds` | 51 | Number of seeds per config (experiment.py). |
| `--variant` | A3 | One of `A1 A2 A3 B C1 C2` (main_gp.py). |
| `--use-ml` | off | Enable ML repositioning layer (main_gp.py). |
| `--classifier` | `stuck_classifier.lgb` | Path to the trained model. |
| `--ml-period` | 5 | Iterations between ML repositioning triggers. |
| `--out` | — | JSONL output path for per-seed results (experiment.py). |

---

## Expected wall-clock times

| Step | Laptop (RTX 4070) | A100 node |
|---|---|---|
| Training data gen (500 runs) | 1–2 h | ~45 min |
| Classifier training | 10–30 min | ~5 min |
| Single benchmark run (10D, one seed) | ~10 s | ~5 s |
| Full 10D benchmark (51 seeds × 12 configs × 5 funcs) | ~1–3 h | ~20 min |
| Full 50D benchmark | ~8–12 h | ~1 h (as array job) |

---

## Troubleshooting

**`RuntimeError: CUDA out of memory`**
Reduce `--dim` or `n_particles`, or cap `MemoryManager(cap=...)` lower. The workload is tiny so this usually means something else is using the GPU — check `nvidia-smi`.

**LightGBM "cannot find GPU" / CUDA driver mismatch**
The wrapper in `stuck_classifier.py` falls back to CPU automatically with a warning. Training on CPU takes ~5× longer but works. To use GPU: ensure LightGBM was built with CUDA support (`python -c "import lightgbm; print(lightgbm.__version__)"` ≥ 4.3, and your CUDA driver is ≥ R525).

**Low test AUC (< 0.7) after training**
Either the training dataset is too small (raise `--runs`) or the label threshold is miscalibrated for your landscape set. Check the reported `pos_rate` per run — if it's near 0 or 1 across the board, adjust `label_thresh` in `generate_training_data.py`.

**GPyTorch GP fit produces NaNs**
Usually a conditioning issue at small memory size. Raise the jitter in `GPSurrogate._posterior` via `gpytorch.settings.cholesky_jitter(1e-4)` context, or extend `fit_iters`.

**WSL2: `torch.cuda.is_available()` is False**
Update the Windows NVIDIA driver (not inside WSL). Restart WSL with `wsl --shutdown` in PowerShell.

---

## Project layout

```
.
├── environment.yml             Conda env spec
├── function.py                 Batched torch benchmark objectives
├── swarm.py                    Tensorized particle swarm
├── gp_surrogate.py             GPyTorch ARD-RBF GP + acquisition
├── memory_manager.py           GP training-set curation
├── features.py                 11-feature extractor for the classifier
├── stuck_classifier.py         LightGBM wrapper (GPU training)
├── ml_repositioner.py          Runtime repositioning layer
├── gp_directed_optimizer.py    Main optimizer class, all variants + ML toggle
├── generate_training_data.py   Offline data-generation pipeline
├── train_classifier.py         Train + evaluate + save
├── experiment.py               Benchmark suite
├── aggregate_results.py        SLURM-array result merger
├── main.py                     Plain-PSO smoke test
├── main_gp.py                  GP-directed PSO smoke test
├── scripts/
│   └── run_local.sh            One-command laptop pipeline
└── slurm/
    ├── generate_data.sbatch
    ├── train.sbatch
    └── experiment.sbatch       Array job template
```

---

## Reference

Original paper: *Yan, Xinyue, and Jun Lu.* "**Gaussian process regression and particle swarm optimization for the optimization of the kinetic parameters.**" (Variant A1–C2 formulations.) [arXiv:2102.04172](https://arxiv.org/abs/2102.04172).

The ML-repositioning layer (classifier + trigger cadence + explore/exploit split) is an extension of the paper, not part of the original method.
