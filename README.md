# pm_env: CayleyPy Implementation Review Benchmark

This repository contains a containerized benchmark for coding agents. The task is to review a buggy implementation of the CayleyPy approach for solving the 3x3x3 Rubik's cube, compare it against the original paper, and fix the implementation bugs that hurt solver quality.

The benchmark is designed to measure more than basic code editing. A strong model needs to:
- inspect and use a local research paper
- distinguish implementation semantics from hyperparameter differences
- reason about architecture, data generation, and inference together
- avoid low-value cleanup that does not address the true performance gap

## What the environment does

The task lives in `env_data/cayleypy_review/`. Inside that directory, the model is given:
- `paper/cayleypy.pdf`: the original CayleyPy paper
- `pilgrim/model.py`: ResMLP architecture
- `pilgrim/trainer.py`: random walk generation and training loop
- `pilgrim/searcher.py`: beam-search style inference
- `pilgrim/utils.py`: helpers
- `train.py` and `test.py`: entry points
- fixed move definitions, target state, and evaluation scrambles

The model is told that the implementation trains and runs, but underperforms relative to the paper. Its job is to identify and fix the high-impact implementation mismatches.

## What is being tested

This benchmark is intentionally not a pure "make tests pass" task. The planted issues are silent semantic bugs that preserve basic executability while degrading solver performance.

The task tests whether an agent can:
- read and use a local PDF as a source document
- keep the paper's algorithmic semantics separate from the task's smaller reference configuration
- locate subtle bugs in:
  - residual wiring
  - random-walk generation
  - training data refresh behavior
  - inference precision / runtime configuration
- resist over-fixing unrelated code smells

## Task framing

The task prompt in [src/pm_env/tasks.py](src/pm_env/tasks.py) makes an important distinction:

- The paper is the reference for algorithmic semantics:
  - model structure and residual wiring
  - random-walk data generation rules
  - inference/search behavior
- The benchmark uses a smaller reference configuration for feasibility:
  - `hd1=1024`
  - `hd2=256`
  - `nrd=1`

So the agent should not treat differences from the paper's larger training budget or Table II hyperparameters as bugs by themselves.

## How grading works

The final score has two parts:

1. Structural witnesses: 30%
- One witness per issue category
- Partial credit is additive

2. Solver evaluation: 70%
- The judge trains the submitted pipeline
- Then it evaluates on 16 held-out DeepCubeA-hard scrambles
- Score is:

```text
solve_rate * mean_optimality
```

where:
- `solve_rate = solved_count / 16`
- `mean_optimality = optimal_length / predicted_solution_length`, averaged over solved cases

Hard failures score zero if the model:
- modifies immutable files
- crashes training or inference
- breaks the lightweight target configuration
- uses a classical Rubik's cube solver

## Hardware

This environment targets GPU evaluation and declares `h100` in the task metadata.

Local iteration can be done on CPU with short runs, but that is only for smoke testing. The real evaluation assumes GPU training and is the authoritative score.

## Setup

### 1. Install `uv`

Follow:

```text
https://docs.astral.sh/uv/getting-started/installation/
```

### 2. Install a container runtime

Use either:
- Podman
- Docker

If using Podman, initialize and start the machine first.

### 3. Create an API key

The default examples below use Anthropic models. Create a key and export it in your shell.

### 4. Sync the local Python environment

```bash
uv sync
```

## Container dependencies

The task requires local PDF extraction so that agents can inspect `paper/cayleypy.pdf` from inside the task container. The container installs `poppler-utils`, which provides `pdftotext`.

This dependency is installed in [Containerfile](Containerfile).

If you modify `Containerfile`, rebuild by rerunning the task after clearing any stale cached image if needed.

## Running the benchmark

### Create a run config

```bash
uv run pm_env create-run-config --model claude-haiku-4-5-20251001 --model-api-key $ANTHROPIC_API_KEY
```

Then edit `run_config.json` and set the task id to:

```text
cayleypy-implementation-review
```

### Run the task

```bash
uv run pm_env run --config run_config.json
```

Transcripts are written to `out/`.

Use `--runtime docker` if you want Docker instead of Podman.

### Run in parallel

```bash
uv run pm_env run --config run_config.json --n-parallel 3
```

## Verifying PDF access manually

Before running the benchmark, it is useful to confirm that PDF extraction works inside the environment:

```bash
cd env_data/cayleypy_review
pdftotext -layout paper/cayleypy.pdf - | head -40
```

You can also extract the full paper text:

```bash
pdftotext -layout paper/cayleypy.pdf paper_text.txt
wc -l paper_text.txt
```

In one recent run, this produced a `paper_text.txt` with 785 lines, confirming that the paper was accessible to the agent.

## Files that define the benchmark

- [src/pm_env/tasks.py](src/pm_env/tasks.py): task prompt and judge wiring
- [Containerfile](Containerfile): container environment
- [src/pm_env/scoring_script.py](src/pm_env/scoring_script.py): general scoring script entry point
- `scoring_data/cayleypy_review/score_cayleypy.py`: task-specific judge
- `env_data/cayleypy_review/`: task assets visible to the model

## Baselines and current observations

### Clean vs buggy calibration

On a small local CPU calibration using a reduced setup, the clean implementation outperformed the buggy implementation:

| Metric | Buggy | Clean |
| --- | ---: | ---: |
| Solved | 3/4 (75%) | 4/4 (100%) |
| Average length on solved | 38.0 | 32.25 |
| Approx. optimality vs length 20 | 0.53 | 0.62 |
| Solver score | 0.40 | 0.62 |

This is not the official benchmark configuration, but it suggests the planted bugs create a real performance gradient even under weak local settings.

### Claude Haiku run

After enabling `pdftotext` in the container and tightening the task prompt, Claude Haiku successfully read the paper but still missed most of the planted bugs.

Observed result from `out/transcript.json`:
- `bug1_skip_connection`: pass
- `bug2_non_backtracking`: fail
- `bug3_data_refreshes`: fail
- `bug4_fp16_inference`: inconclusive on CPU-only witness
- `solve_rate`: 0.625
- `mean_optimality`: 0.657
- `solver_score`: 0.411
- `final_score`: 0.310

This is a useful benchmark outcome:
- the agent can now access the paper
- failure is no longer explained by missing PDF tooling
- the remaining miss appears to be a model capability / reasoning issue rather than an environment issue

## Why the PDF matters

The benchmark is intentionally shaped so that code-only inspection is not enough. The most important bugs are easier to identify when the implementation is cross-checked against the paper's description and figures.

That is why the environment now:
- instructs the agent to inspect the paper before editing
- provides `pdftotext` inside the container

## Interpreting scores

This benchmark should not be judged by whether a small baseline model gets 100%. A better sign is:
- weaker models struggle
- stronger models do better
- frontier models do substantially better but do not instantly saturate
- failure cases remain interpretable

The next planned comparison is to evaluate stronger models beyond Claude Haiku, including Gemini-based runs.

## AI usage

AI tools were used during benchmark development for:
- inspecting transcripts
- evaluating whether paper access worked
- refining the task prompt wording
- improving the benchmark documentation

The benchmark design itself, planted bugs, and scoring intent were authored manually.
