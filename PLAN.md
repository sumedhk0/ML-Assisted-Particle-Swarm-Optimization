# PLAN

## Objective

Update the project objective to:

> determine when learned particle-level rescue is worth its computational
> overhead in surrogate-guided PSO, and whether it outperforms random or
> heuristic rescue under fixed evaluation and time budgets on high-dimensional
> expensive black-box problems

This changes the project from a pure "final objective value is lower" benchmark
into a controlled cost-benefit study.

---

## Current status

What the codebase already does:

- trains a LightGBM classifier to detect likely stuck particles
- applies learned particle-level rescue on top of GP-guided PSO
- benchmarks vanilla `PSO`, GP variants `A1-A3/B/C1/C2`, and `+ML` variants
- records final per-seed objective value
- now records per-run wall-clock, iteration count, evaluation count, optional
  target-hit metrics, and lightweight convergence curves for new runs

What the codebase does **not** yet do:

- compare learned rescue against random rescue
- compare learned rescue against a hand-designed heuristic rescue
- run under explicit wall-clock budgets
- isolate rescue overhead from the rest of the GP-guided optimizer
- benchmark truly expensive black-box objectives in a controlled way

---

## Main research question

The key question is no longer just:

> does learned rescue improve final quality?

It is now:

> when does the extra compute from learned rescue pay off, and is the learned
> selector better than simpler rescue policies under matched budgets?

That means every future comparison must be fair on both:

- **optimization budget**
- **computational overhead**

---

## What data we are missing

### 1. Matched rescue-policy baselines

We currently have:

- no rescue
- learned rescue

We are missing:

- `random rescue`
- `heuristic rescue`

Without those, we cannot tell whether the learned selector itself matters, or if
any occasional rescue would have worked.

### 2. Matched wall-clock budget runs

We currently compare methods mostly at a fixed evaluation budget.

We are missing:

- stop-at-time-budget runs
- time-to-target comparisons
- hit-rate under wall-clock cutoffs

This is necessary because learned rescue adds compute overhead per iteration.

### 3. Rescue-overhead decomposition

We currently have total per-run wall time.

We are missing per-iteration or per-component timing for:

- GP fit time
- feature extraction time
- LightGBM inference time
- rescue target search time
- particle reset time

Without this, we can say "ML is slower" but not **why**.

### 4. Expensive-objective sensitivity data

The current benchmarks use standard analytic functions. They are useful, but
they are still cheap to evaluate relative to many real black-box objectives.

We are missing controlled experiments where objective evaluations are made more
expensive, so we can answer:

- does rescue overhead become negligible when the black-box is expensive?
- at what objective-cost scale does learned rescue become worthwhile?

### 5. Fresh convergence data for older completed runs

The older completed `sphere`, `rastrigin`, and `ackley` runs were finished
before the new convergence / time-to-target instrumentation existed.

We are missing:

- convergence curves for those historical runs
- target-hit metrics for those historical runs

This means the current efficiency analysis can only be done properly on new runs
unless we rerun those problems.

### 6. High-dimensional sweep beyond one setting

Right now the main benchmark setting is `dim=50`.

We are missing a systematic sweep over:

- `dim=50`
- `dim=100`
- possibly `dim=200`

This matters because rescue may help more as dimensionality increases and the GP
becomes less reliable.

### 7. Statistical significance / uncertainty summaries

We currently summarize medians and means.

We are missing:

- bootstrap confidence intervals
- paired significance tests across seeds
- effect-size reporting

This is needed for advisor- and paper-grade claims.

---

## Fair comparison principles

To answer the new objective cleanly, we should compare policies under **matched
rescue mechanics** and vary only the **selector**.

That means keeping these fixed:

- same base optimizer
- same `ml_period`
- same `top_k_frac`
- same exploit / explore split
- same GP target generators
- same jitter policy
- same evaluation budget
- same wall-clock budget

Only the rescue selector changes.

This is the cleanest way to attribute gains to "learned particle-level rescue"
rather than to the relocation target design itself.

---

## Rescue policies to implement

### Policy 0: no rescue

This is the current baseline GP-guided run.

### Policy 1: learned rescue

This is the current LightGBM-based selector.

### Policy 2: random rescue

Select the same number of particles as the learned policy, but choose them
uniformly at random from the eligible particles.

Important control:

- keep the same exploit / explore destination rules
- keep the same rescue period and top-k count

This isolates the value of **which particles** are rescued.

### Policy 3: heuristic rescue

Use a hand-designed score built from existing swarm state, for example:

- highest `pbest_plateau`
- lowest recent improvement
- high distance from `gbest`

Recommended first heuristic:

- rank by plateau length, break ties by worst current value

This is simple, interpretable, and already supported by existing features.

### Optional Policy 4: heuristic-GP uncertainty rescue

Same heuristic particle selection, but target particles using uncertainty-aware
relocation. This is useful if we want to separate selection quality from target
quality further.

---

## Implementation plan

### Workstream A: refactor rescue into policy objects

Goal:

- make `learned`, `random`, and `heuristic` rescue interchangeable

Status:

- implemented in code
- `experiment.py` and `main_gp.py` now accept rescue-policy selection
- current available policies are `none`, `learned`, `random`, and
  `heuristic_plateau`
- next validation step is compute-node smoke testing plus fixed-budget
  comparisons against the existing completed baselines

Tasks:

1. Introduce a shared rescue-policy interface.
2. Rename the current `MLRepositioner` into a learned-policy implementation or
   wrap it under a policy dispatcher.
3. Add `RandomRepositioner`.
4. Add `HeuristicRepositioner`.
5. Pass a `--rescue-policy` argument through `main_gp.py`, `experiment.py`, and
   Slurm scripts.

Recommended enum values:

- `none`
- `learned`
- `random`
- `heuristic_plateau`

### Workstream B: add matched-budget support

Goal:

- compare policies under both evaluation budgets and wall-clock budgets

Status:

- implemented in code
- `experiment.py`, `main.py`, and `main_gp.py` now accept
  `--max-wall-time-sec`
- both GP-guided PSO and vanilla `PSO` now stop at the earlier of:
  - evaluation-budget exhaustion
  - wall-clock-budget exhaustion
- per-run outputs now record the stop reason
- current implementation enforces wall-clock limits at iteration / stage
  boundaries; non-preemptive GP fit can still overshoot very small time budgets
- next step is to run matched-budget experiments and compare hit-rate and
  time-to-target under fixed time limits

Tasks:

1. Add `--max-wall-time-sec` to the optimizer / experiment path.
2. Stop runs when either:
   - evaluation budget is exhausted
   - wall-clock budget is reached
3. Record whether the stop was budget-limited by evaluations or time.

### Workstream C: add timing decomposition

Goal:

- explain where the overhead comes from

Status:

- implemented in code at the aggregated per-run level
- per-run outputs now record totals for:
  - GP fit time
  - feature extraction time
  - classifier inference time
  - acquisition search time
  - rescue reset time
  - rescue event counts
  - rescued particle counts
- convergence-curve NPZ files now carry the same aggregate timing metadata
- next step is compute-node smoke testing plus real runs that use the new
  fields in analysis tables

Tasks:

1. Add timers around:
   - GP fit
   - feature extraction
   - classifier inference
   - acquisition search
   - rescue reset
2. Save aggregated timing totals per run.
3. Optionally save per-iteration timing summaries in convergence NPZs.

Recommended JSONL fields:

- `time_gp_fit_sec`
- `time_feature_sec`
- `time_inference_sec`
- `time_acquisition_sec`
- `time_rescue_reset_sec`
- `n_rescue_events`
- `n_particles_rescued`

### Workstream D: add expensive-objective benchmark mode

Goal:

- test whether rescue overhead is worth it when objective evaluations are
  expensive

Simplest implementation:

1. Add an objective wrapper that injects a controlled delay per evaluation.
2. Support values like:
   - `0 ms`
   - `1 ms`
   - `5 ms`
   - `10 ms`
3. Apply the same delay to all methods.

Alternative implementation:

- repeat function computation multiple times instead of sleeping, if CPU/GPU
  utilization realism matters more than pure wall-clock emulation

### Workstream E: analysis scripts

Goal:

- generate advisor-ready tables and plots automatically

Tasks:

1. Add a script to aggregate:
   - median final objective
   - median wall-clock
   - hit-rate for target thresholds
   - median time-to-target
   - median evaluations-to-target
2. Add convergence-curve plotting:
   - best value vs evaluations
   - best value vs seconds
3. Add cost-normalized metrics:
   - improvement per CPU-hour
   - improvement per GPU-hour
4. Add comparison tables:
   - learned vs random
   - learned vs heuristic
   - no rescue vs learned

---

## Recommended experimental matrix

### Phase 1: selector isolation

Use strong base variants first:

- `B`
- `C1`
- `C2`

Compare:

- no rescue
- learned rescue
- random rescue
- heuristic rescue

Problems:

- `sphere`
- `rastrigin`
- `ackley`
- `griewank`
- `rosenbrock`

Dimension:

- start with `50`

Budgets:

- current fixed evaluation budget
- matched wall-clock budget

### Phase 2: expensive-objective sensitivity

Use the strongest two base families from Phase 1.

Objective-cost levels:

- no delay
- low delay
- medium delay
- high delay

Question:

- at what objective cost does learned rescue become worth its overhead?

### Phase 3: dimensionality sweep

Dimensions:

- `50`
- `100`
- `200`

Question:

- does learned rescue become more valuable as search dimensionality increases?

---

## Metrics to report

### Quality metrics

- median final objective
- mean final objective
- best achieved objective

### Efficiency metrics

- median wall-clock per run
- median evaluations used
- median iterations used
- median time-to-target
- median evaluations-to-target
- target hit-rate

### Overhead metrics

- rescue compute fraction of total runtime
- GP fit fraction of total runtime
- classifier inference fraction of total runtime
- one-time training cost

### Amortization metrics

Report two versions:

1. **online-only**
   - excludes one-time classifier training cost
2. **amortized**
   - includes a share of data generation and classifier training cost

This matters because the learned policy has upfront offline cost that random and
heuristic policies do not.

---

## Statistical analysis plan

For each problem / dimension / policy:

- report median and interquartile range
- bootstrap confidence intervals for medians
- paired comparisons across matched seeds
- Wilcoxon signed-rank or paired bootstrap differences

Primary decision rule:

- learned rescue is "worth it" only if it improves quality or hit-rate enough
  to justify its added runtime under the target budget regime

---

## Minimal next implementation steps

### Step 1

Refactor rescue into a policy interface and add:

- `random`
- `heuristic_plateau`

### Step 2

Add per-component timing instrumentation around rescue-related code.

### Step 3

Add wall-clock stopping support.

### Step 4

Rerun:

- `sphere`
- `rastrigin`
- `ackley`

with the new instrumentation so they are comparable to the newer runs.

### Step 5

Run the selector-isolation study on `B`, `C1`, and `C2`.

---

## Success criteria

This project should be considered successful if we can answer all of the
following with data:

1. Does learned rescue beat no rescue in final quality?
2. Does learned rescue beat random rescue under the same budget?
3. Does learned rescue beat heuristic rescue under the same budget?
4. How much extra runtime does learned rescue cost?
5. When objective evaluations become expensive, does that overhead become
   negligible?
6. Under which problems and dimensions is learned rescue actually worth it?

---

## Risks

- learned rescue may improve final quality but lose badly on wall-clock
- random or heuristic rescue may perform nearly as well, reducing the value of
  the classifier
- the strongest conclusion may be conditional, for example:
  - worth it only on expensive objectives
  - worth it only for `B` / `C1`
  - not worth it for `A*`

That is still a valid scientific outcome.

---

## Deliverables

- updated benchmark runner with rescue-policy controls
- updated JSONL schema with timing decomposition
- convergence and time-to-target plots
- learned vs random vs heuristic comparison tables
- advisor-ready summary answering:
  - when is learned rescue worth it?
  - when is it not?
