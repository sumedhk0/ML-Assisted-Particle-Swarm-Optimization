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

## Related work and novelty

This repository is best understood as an extension of prior GP-guided PSO work,
not as an attempt to claim that every ingredient is individually new.

### Closest prior work

The six `A1`-`C2` GP-directed variants implemented here come directly from:

- Johannes Jakubik, Adrian Binding, and Stefan Feuerriegel. 2021.
  *Directed particle swarm optimization with Gaussian-process-based function
  forecasting*. arXiv:2102.04172.
  https://arxiv.org/abs/2102.04172

That paper introduces the central idea of fitting a Gaussian-process surrogate to
past particle evaluations and then steering particle motion with GP-derived
search targets.

### Related PSO modification literature

Several adjacent ideas already exist in the literature:

- Shaochun Qu, Fuguang Liu, and Zijian Cao. 2024.
  *An Adaptive Surrogate-Assisted Particle Swarm Optimization Algorithm
  Combining Effectively Global and Local Surrogate Models and Its Application*.
  *Applied Sciences* 14(17):7853.
  https://doi.org/10.3390/app14177853
- Stephen Chen, Imran Abdulselam, Naeeme YadollahPour, and
  Yasser González-Fernández. 2021.
  *Stall Detection in Particle Swarm Optimization*.
  CEC 2021 workshop paper.
  https://www.yorku.ca/sychen/research/workshops/CEC2021_Workshop_on_Selection_PSO.pdf
- Vilmar Steffen. 2022.
  *Particle Swarm Optimization with a Simplex Strategy to Avoid Getting Stuck on
  Local Optimum*. IntechOpen.
  https://doi.org/10.5772/acrt.11

These works show that:
- surrogate-assisted PSO is an established direction
- stall / stagnation detection is an established direction
- targeted particle repositioning to escape local optima is also established

### What is new in this repository

To the best of our knowledge, the specific combination implemented here is not a
standard public baseline:

1. The GP-directed motion rules from Jakubik et al. are preserved as the core
   optimizer family (`A1`-`C2`).
2. A separate offline-trained LightGBM classifier predicts which particles are
   likely to be stuck or unproductive during the run.
3. Every `ml_period` iterations, the highest-risk particles are repositioned
   using two GP-derived targets:
   - the GP lower-confidence-bound minimum for exploitative relocation
   - the GP maximum-uncertainty point for exploratory relocation
4. The entire stack is packaged as a tensorized PyTorch / GPyTorch
   implementation with reproducible SLURM workflows, vanilla-PSO reference
   runs, and trajectory trace / animation support.

In other words, the novelty here is the hybrid policy:

> GP-directed PSO + particle-level ML stagnation detection +
> targeted GP-based exploit/explore relocation

rather than the isolated invention of GP surrogates, PSO, or particle
repositioning by themselves.

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
git clone https://github.com/sumedhk0/ML-Assisted-Particle-Swarm-Optimization.git
cd "ML-Assisted-Particle-Swarm-Optimization"
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
git clone https://github.com/sumedhk0/ML-Assisted-Particle-Swarm-Optimization.git ~/pso
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
git clone https://github.com/sumedhk0/ML-Assisted-Particle-Swarm-Optimization.git
cd "ML-Assisted-Particle-Swarm-Optimization"

# Load modules (edit slurm/*.sbatch to match your cluster's module system)
module load anaconda/2024.02   # or whatever's available
module load cuda/12.1

conda env create -f environment.yml
conda activate pso-gpu
```

Then edit the three `slurm/*.sbatch` files: uncomment the `module load` lines, adjust `--time`, `--cpus-per-task`, partition/account flags as needed by your cluster.

### macOS / CPU-only

```bash
git clone https://github.com/sumedhk0/ML-Assisted-Particle-Swarm-Optimization.git
cd "ML-Assisted-Particle-Swarm-Optimization"
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

# 3b. Smoke test a non-learned rescue policy
python main_gp.py --dim 10 --variant B --rescue-policy heuristic_plateau --device cpu

# 4. Benchmark (51 seeds × 6 variants × 5 functions × ML on/off)
python experiment.py --dim 10 --seeds 51 --device cuda:0 \
    --out out/results_dim10.jsonl

# 5. Print summary
python aggregate_results.py out/results_dim10.jsonl
```

### Trajectory traces for animation

You can save per-iteration particle positions for later animation:

```bash
python main.py --dim 2 --func sphere --iters 200 --trace-out traces/pso_sphere_seed0.npz

python main_gp.py --dim 2 --func rastrigin --variant C1 --use-ml \
    --trace-out traces/gp_c1_ml_seed0.npz --trace-every 2
```

For benchmark or SLURM runs, use `experiment.py`:

```bash
python experiment.py --functions sphere --variants PSO,C1 --baseline-only \
    --dim 2 --seeds 3 --device cpu --trace-dir traces --trace-every 2

python experiment.py --functions ackley --variants B,C1 \
    --rescue-policies none,learned,random,heuristic_plateau \
    --dim 50 --seeds 5 --device cpu --out out/ackley_rescue_compare.jsonl
```

Each trace file stores particle positions, personal-best positions, current values,
global-best positions, and the iteration/evaluation counters. Render a saved trace to
GIF with:

```bash
python render_trace_animation.py traces/trace_sphere_dim2_PSO_seed0.npz
```

For dimensions greater than 2, the renderer automatically uses a PCA projection.

### Convergence curves and time-to-target metrics

For efficiency studies, you usually want lighter-weight convergence data than the
full particle-position traces. `experiment.py` can now save per-seed best-so-far
curves and record first-hit metrics for objective thresholds of interest:

```bash
python experiment.py --functions ackley --variants PSO,B,C1 --dim 50 --seeds 3 \
    --device cpu --baseline-only --out out/ackley_small.jsonl \
    --curve-dir curves --targets 1.0,0.5 --max-wall-time-sec 30
```

This adds the following fields to each JSONL row:
- `wall_time_sec`
- `eval_count`
- `n_iterations`
- `stop_reason`
- `time_gp_fit_sec`
- `time_feature_sec`
- `time_inference_sec`
- `time_acquisition_sec`
- `time_rescue_reset_sec`
- `n_rescue_events`
- `n_particles_rescued`
- `target_hits` (when `--targets` is provided)

and writes one compact NPZ file per seed to `--curve-dir`, containing:
- `best_values`
- `iterations`
- `eval_counts`
- `elapsed_times_sec`

Those curve files are the recommended source for:
- anytime convergence plots
- time-to-target comparisons
- evaluations-to-target comparisons
- quality-vs-wall-clock tradeoff analysis

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
| `--rescue-policy` | `none` / `learned` via `--use-ml` | Override the particle rescue selector in `main_gp.py`. |
| `--rescue-policies` | — | Comma-delimited rescue policies to benchmark in `experiment.py`. |
| `--classifier` | `stuck_classifier.lgb` | Path to the trained model. |
| `--ml-period` | 5 | Iterations between ML repositioning triggers. |
| `--out` | — | JSONL output path for per-seed results (experiment.py). |
| `--trace-out` | — | Save one optimizer run as a compressed trajectory NPZ (`main.py`, `main_gp.py`). |
| `--trace-dir` | — | Save one NPZ trace per seed (`experiment.py`). |
| `--trace-every` | 1 | Record every k-th iteration when tracing. |
| `--curve-out` | — | Save one lightweight best-so-far convergence NPZ (`main.py`, `main_gp.py`). |
| `--curve-dir` | — | Save one convergence NPZ per seed (`experiment.py`). |
| `--max-wall-time-sec` | — | Stop a run early when the wall-clock budget is exhausted. |
| `--targets` | — | Comma-delimited objective thresholds for first-hit metrics. |

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

## Benchmark snapshot

The tables below summarize the current `dim=50`, `51`-seed runs on the PACE
cluster. Values are the median final objective value across seeds, so lower is
better.

### Sphere (complete)

| Variant | Median |
|---|---:|
| A1 | 80.805 |
| A2 | 79.519 |
| A3 | 87.405 |
| B | 1.178 |
| C1 | 1.132 |
| C2 | 20.272 |
| A1+ML | 81.930 |
| A2+ML | 80.779 |
| A3+ML | 87.135 |
| B+ML | 0.492 |
| C1+ML | 0.475 |
| C2+ML | 0.901 |

Main takeaways:
- `C1+ML` is the best sphere configuration so far.
- `B+ML` is a close second.
- `C2` improves dramatically once ML repositioning is enabled.
- The `A*` family remains much worse than the rescue-style variants.

### Rastrigin (complete)

| Variant | Median |
|---|---:|
| PSO | 201.500 |
| A1 | 496.050 |
| A2 | 490.194 |
| A3 | 499.578 |
| B | 170.762 |
| C1 | 151.477 |
| C2 | 203.639 |
| A1+ML | 495.594 |
| A2+ML | 492.889 |
| A3+ML | 495.607 |
| B+ML | 95.098 |
| C1+ML | 93.443 |
| C2+ML | 97.538 |

Main takeaways:
- Vanilla `PSO` is a meaningful baseline and outperforms plain `C2`, but not
  the stronger GP-guided variants.
- `C1+ML`, `B+ML`, and `C2+ML` all beat vanilla `PSO` comfortably.
- The `A*` family is consistently poor on this landscape.

### Ackley (complete)

| Variant | Median |
|---|---:|
| PSO | 2.891 |
| A1 | 6.106 |
| A2 | 6.069 |
| A3 | 6.186 |
| B | 0.570 |
| C1 | 0.724 |
| C2 | 3.958 |
| A1+ML | 6.181 |
| A2+ML | 6.138 |
| A3+ML | 6.240 |
| B+ML | 0.278 |
| C1+ML | 0.344 |
| C2+ML | 1.584 |

Main takeaways:
- `B+ML` is the best ackley configuration so far.
- `C1+ML` is also very strong, and both beat vanilla `PSO` comfortably.
- Vanilla `PSO` still beats the `A*` family, which remains consistently weak.

### Griewank (complete)

| Variant | Median |
|---|---:|
| PSO | 15.635 |
| A1 | 290.329 |
| A2 | 276.347 |
| A3 | 287.785 |
| B | 65.698 |
| C1 | 67.468 |
| C2 | 67.468 |
| A1+ML | 287.562 |
| A2+ML | 285.721 |
| A3+ML | 298.494 |
| B+ML | 20.568 |
| C1+ML | 20.317 |
| C2+ML | 20.317 |

Main takeaways:
- Vanilla `PSO` is the best griewank configuration so far.
- `B+ML`, `C1+ML`, and `C2+ML` still improve dramatically over their plain
  GP-guided counterparts, but they do not beat vanilla `PSO`.
- This appears to be a refinement problem rather than a basin-finding problem:
  all of the strong methods reliably reach coarse thresholds, but vanilla `PSO`
  converts that into low final values more often.
- A useful intuition is that griewank's broad basin plus oscillatory cosine
  structure is friendly to standard PSO contraction, while the surrogate-guided
  relocations help coarse progress but can disturb late-stage refinement.

### Rosenbrock (complete)

| Variant | Median |
|---|---:|
| PSO | 3214.229 |
| A1 | 296174.250 |
| A2 | 307323.125 |
| A3 | 291431.438 |
| B | 21484.953 |
| C1 | 21632.215 |
| C2 | 35822.902 |
| A1+ML | 310656.844 |
| A2+ML | 304861.625 |
| A3+ML | 297177.562 |
| B+ML | 3247.840 |
| C1+ML | 2454.871 |
| C2+ML | 2524.152 |

Main takeaways:
- `C1+ML` is the best rosenbrock configuration so far.
- `C2+ML` is a close second, and both beat vanilla `PSO`.
- The learned rescue layer is especially valuable here because it turns the
  weak plain GP-guided variants into methods that are competitive with or
  better than vanilla `PSO` on this narrow-valley landscape.

### Ackley rescue-policy comparison (fixed wall-clock budget, complete)

To answer the newer research question, we ran a matched-budget policy
comparison on `ackley`, `dim=50`, using the strong GP variants `B` and `C1`
with a fixed `240 s` wall-clock budget per seed. This isolates the value of the
particle selector under a real time cap.

Median final objective values:

| Variant | No rescue | Learned | Random | Heuristic plateau |
|---|---:|---:|---:|---:|
| B | 0.657 | 0.647 | 0.600 | 0.667 |
| C1 | 0.805 | 0.782 | 0.613 | 0.826 |

Selected target-hit rates under the same `240 s` budget:

| Variant | Policy | Hit rate `<= 1.0` | Hit rate `<= 0.5` |
|---|---|---:|---:|
| B | none | 0.941 | 0.059 |
| B | learned | 0.961 | 0.118 |
| B | random | 0.922 | 0.255 |
| B | heuristic | 0.980 | 0.098 |
| C1 | none | 0.745 | 0.039 |
| C1 | learned | 0.863 | 0.059 |
| C1 | random | 0.941 | 0.235 |
| C1 | heuristic | 0.686 | 0.039 |

Main takeaways:
- Under a fixed wall-clock budget, `random rescue` is currently the strongest
  policy for both `B` and `C1`.
- `learned rescue` still beats `no rescue`, but its extra overhead matters much
  more once the budget is time-limited.
- This is an important counterpoint to the fixed-evaluation results, where
  learned rescue is usually strongest on final quality.

Current caveat:
- The first fixed-evaluation policy-comparison attempt used seed chunks that
  were too large and several jobs hit the Slurm time limit, so those numbers
  should be treated as partial only. A smaller-chunk rerun is the right source
  for the final fixed-evaluation comparison.

---

## Compute cost, time-to-target, and efficiency

These are the advisor-facing questions we should answer explicitly.

### Are we using more or less compute than the baselines?

At the workflow level, we are now using cluster resources much more sensibly:
- one-time training-data generation uses the GPU
- classifier training uses CPU
- benchmark experiments use CPU

That is a real improvement over the earlier all-GPU setup.

At the algorithm-comparison level, however, the current ML-enhanced variants are
generally **more expensive in wall-clock time per benchmark chunk** than their
non-ML counterparts.

Average CPU chunk runtime on the completed `dim=50` runs:

| Problem | Baseline avg | ML avg | ML overhead |
|---|---:|---:|---:|
| Sphere | 2.08 h | 2.33 h | +11.9 % |
| Rastrigin | 3.80 h | 4.66 h | +22.6 % |
| Ackley | 4.02 h | 4.16 h | +3.5 % |

One-time preprocessing cost per problem:
- training-data generation: about `3.0-3.3 GPU-hours`
- classifier training: about `0.02-0.19 CPU-hours`

So the honest current claim is:
- **better objective quality**
- **better GPU hygiene**
- **slightly to moderately higher compute cost**

### Why are some of our methods more expensive?

There are three main reasons:

1. The GP-guided variants fit or refit a surrogate model during the run, while
   vanilla `PSO` does not.
2. The `B`, `C1`, and `C2` families perform additional acquisition searches to
   decide where to relocate particles.
3. The `+ML` variants add classifier inference and extra repositioning work
   every `ml_period` iterations.

In short, we are paying extra compute per iteration for better guidance.

### Are we actually getting enough benefit to justify that extra cost?

For the strong variant families, yes on objective quality.

Representative median improvements from baseline to `+ML`:
- `sphere`: `B -> B+ML` improves by `58.3 %`
- `sphere`: `C1 -> C1+ML` improves by `58.1 %`
- `rastrigin`: `B -> B+ML` improves by `44.3 %`
- `rastrigin`: `C1 -> C1+ML` improves by `38.3 %`
- `ackley`: `B -> B+ML` improves by `51.2 %`
- `ackley`: `C1 -> C1+ML` improves by `52.6 %`

So the current evidence supports:

> modest runtime overhead in exchange for large quality gains

rather than:

> same quality with lower compute

### Can we already claim better time to global optimum?

Not yet.

Two reasons:
- In these `50D` runs, most methods do **not** reliably hit the exact optimum
  `0`, so "time to global optimum" is often undefined.
- Older benchmark outputs only stored the final best value per seed, which was
  enough for quality tables but not for convergence-time analysis.

The more defensible metric is **time to target value**, not time to exact
optimum.

### What should we report going forward?

The recommended efficiency metrics are:

1. **Time to target value**
   - Example thresholds:
   - `sphere`: first hit below `1.0`
   - `ackley`: first hit below `0.5`
   - `rastrigin`: first hit below `100`
2. **Evaluations to target**
3. **Anytime convergence curves**
4. **Objective improvement per CPU-hour / GPU-hour**

### What is already instrumented in the code now?

Future benchmark runs can now record:
- per-seed `wall_time_sec`
- per-seed `eval_count`
- per-seed `n_iterations`
- optional `target_hits` via `--targets`
- optional lightweight convergence NPZs via `--curve-dir`

That means future runs can support:
- median wall-clock by method
- hit-rate for threshold targets
- median time-to-target
- median evaluations-to-target
- best-so-far vs seconds / evaluations plots

### Bottom line

Right now, the strongest defensible statement is:

- the best `+ML` variants (`B+ML`, `C1+ML`, often `C2+ML`) are **substantially
  better in final objective quality**
- they are **not yet cheaper in raw compute time**
- the workflow is **much more disciplined in how it uses GPU resources**
- the codebase is now instrumented to measure efficiency properly on the next
  run set

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
├── trace_utils.py              NPZ trace recorder for per-iteration swarm states
├── render_trace_animation.py   Trace-to-GIF renderer (raw 2D or PCA projection)
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
