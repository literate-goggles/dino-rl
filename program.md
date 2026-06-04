# autoresearch

This experiment asks the LLM to **speed up training** of the **browser-image PPO**
agent for Chrome Dino.

## Context

The repo now has two distinct PPO tracks:

| Track | Backend | Observation | Status |
|---|---|---|---|
| Feature PPO | Python simulator | 10 engineered features | Strong; already far ahead |
| Image PPO | Real Chrome + ChromeDriver | 4 x 84 x 84 grayscale stack | Current research target |

This file is about the **image PPO** track only.

Current browser-image setup:

- Environment backend: real `chrome://dino` through `dino_rl/browser_env.py`
- Observation: 4 stacked grayscale frames of shape `(4, 84, 84)`
- Preprocessing: crop to gameplay strip, mask score HUD, max-pool recent frames
- Action repeat: `4`
- Action space: `3` actions
  - `0 = noop`
  - `1 = jump`
  - `2 = duck / hold down`
- Reward:
  - positive reward from scaled `distanceRan` progress
  - `-1.0` on crash
- Browser recovery:
  - the env is expected to recover from dead sessions / tab crashes instead of
    killing the trainer

## Goal

The objective of this program is to **make training faster**, not to push eval
quality higher. Concretely, optimize the trainer for throughput and wall-clock
efficiency:

- **primary target: maximize environment steps per second (`steps/sec`)** sustained
  over a 1-hour run
- secondary target: minimize wall-clock time per PPO update (`sec/update`)
- secondary target: maximize total updates and total env steps completed within a
  fixed 1-hour budget

Treat eval score as a **guardrail, not the objective**. A speed change is only
valid if learning does not visibly collapse: the run should still trend upward and
stay in the same ballpark as the current baseline (`eval avg ~75`). Do not trade
away all learning for raw throughput — a fast trainer that learns nothing is a
failed experiment. But within that guardrail, **always prefer the faster option.**

The bottleneck here is almost entirely the browser/env interaction loop, not the
GPU. This is a systems-and-throughput problem, not a sample-efficiency problem.

## Setup

To start a new research pass, work with the user to:

1. Propose a run tag based on the date, for example `apr20-image`
2. Create a fresh branch `autoresearch/<tag>` from current `master`
3. Read the in-scope files:
   - `dino_rl/algorithms/ppo.py`
   - `dino_rl/browser_env.py`
   - `dino_rl/play_browser.py`
   - `dino_rl/policy_loader.py`
4. Confirm that Chrome/ChromeDriver browser control is functional
5. Establish a **baseline `steps/sec`** for the current code before changing anything
6. Kick off the experiment loop

## Experimentation

Each experiment uses a single GPU and runs for a **fixed wall-clock budget of 1
hour**. The point of the fixed budget is to compare how much training work each
variant gets done in the same amount of time.

**Always run on GPU 2.** Pin every training, eval, and profiling run to that
device with `CUDA_VISIBLE_DEVICES=2` (as in the commands below). Do not use any
other GPU — the other devices may be in use by other work.

Launch command from repo root:

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m dino_rl.algorithms.ppo \
  --env-backend browser \
  --observation-mode image \
  --time-budget-sec 3600 \
  --eval-every 5 \
  --print-every 1 \
  > run_browser_image.log 2>&1
```

If a best browser-image checkpoint already exists, resume from it unless there
is a good reason not to:

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m dino_rl.algorithms.ppo \
  --env-backend browser \
  --observation-mode image \
  --time-budget-sec 3600 \
  --eval-every 5 \
  --print-every 1 \
  --init-checkpoint checkpoints/dino_ppo_browser_image_best.pth \
  > run_browser_image.log 2>&1
```

### What you CAN modify

- `dino_rl/algorithms/ppo.py`
- `dino_rl/browser_env.py`
- `dino_rl/play_browser.py`
- `dino_rl/policy_loader.py`

Everything that affects training throughput is fair game. Speed levers worth
exploring include:

- browser/env loop: reduce round-trips to ChromeDriver, batch or cache JS calls,
  cut sleeps and polling intervals, tighten the screenshot → preprocess path
- observation pipeline: cheaper crop/mask/resize, fewer copies, vectorized
  preprocessing, faster grayscale/stacking
- parallelism: multiple browser workers / parallel env rollouts feeding one learner
- data movement: fewer host↔GPU transfers, pinned memory, larger contiguous batches
- compute: mixed precision, `torch.compile`, CNN encoder cost, larger minibatches
- PPO loop overhead: rollout length, number of epochs, eval cadence, logging cost
- action repeat and frame-skip tuning to get more env progress per browser step

### What you CANNOT modify

- `dino_rl/env.py` for the purpose of making the browser-image task easier
- install new packages
- silently switch the experiment back to simulator features
- you may not "speed up" by disabling learning, shrinking the rollout to nothing,
  or otherwise gutting the algorithm so it no longer trains

The task here is specifically to make PPO on **browser images** train faster while
still learning.

## Key Technical Facts

- Action space is `3`, so max entropy is `ln(3) ~= 1.099`
- Browser-image observations are expensive; the env step is the dominant cost and
  the main thing to optimize
- Current image defaults:
  - `rollout_len = 512`
  - `minibatch_size = 64`
  - `eval_every = 5`
  - `score_delta_coeff = 0.0`
- The old future auxiliary loss is gone; this image path is now plain PPO with
  a CNN encoder
- Browser failures such as `tab crashed`, disconnected sessions, or dead
  ChromeDriver connections should be treated as recoverable env failures, not as
  acceptable reasons for the trainer to exit. Recovery should also be **fast** —
  slow restarts eat the time budget.

## Output Format

The trainer prints periodic updates like:

```text
Update   40/100000 | Steps   20480 | Episodes   6 | AvgScore    54.3 | PolicyL  0.0012 | ValueL 121.337 | Entropy 0.9812 | KL 0.00234 | Clip 0.154
  >> Eval @ update 40: avg=72.7  min=58  max=96
```

View TensorBoard with:

```bash
tensorboard --logdir results/runs
```

Key metrics to watch (speed first):

- **`steps/sec`** — the target metric; derive it from `Steps` and wall-clock if
  the trainer does not print it directly, and add a throughput log line if needed
- **`sec/update`** — time per PPO update; should go down
- **total `Steps` / total `Update` count at the 1-hour mark** — more is better
- browser recovery messages in the log — slow or frequent recovery kills throughput
- guardrail only: `eval/avg_score` should not collapse versus baseline
- guardrail only: `train/entropy` near `1.099` means the policy went random

## Logging Results

When an experiment finishes, log it to `results.tsv` as tab-separated text:

```text
commit	steps_per_sec	total_steps	eval_avg	status	description
```

Columns:

1. short git commit hash
2. `steps_per_sec` (sustained, primary metric)
3. `total_steps` completed within the 1-hour budget
4. `eval_avg` (guardrail — did learning survive?)
5. status: `keep`, `discard`, or `crash`
6. short description of the speedup attempted

Do not commit `results.tsv`.

## The Experiment Loop

Run on a dedicated branch such as `autoresearch/apr20-image`.

Loop:

1. Inspect the current branch and commit
2. Make one focused **throughput** change
3. Commit the change
4. Run a 1-hour browser-image PPO experiment
5. Read the results from the log: compute `steps/sec` and `total_steps`
6. If the run crashed:
   - inspect the traceback
   - distinguish between a browser-recovery bug, a trainer bug, and a bad idea
7. Record the result in `results.tsv`
8. If `steps/sec` improved **and** eval did not collapse, keep the commit
9. If `steps/sec` did not improve, or learning broke the guardrail, revert the
   change and continue from the last good point

### Profiling first

Before guessing, profile. Identify where wall-clock actually goes (browser
round-trips, screenshot capture, preprocessing, GPU step, optimization) and attack
the biggest cost. Prefer changes justified by a measured bottleneck over
speculative micro-optimizations.

### Timeout rule

- Budget per experiment: `3600` seconds
- If a run goes materially past `1 hour` without the trainer honoring the time
  budget, stop it manually and treat that as a failure in the experiment loop

### Crash rule

- A recoverable Chrome crash should not kill PPO
- If PPO still exits on browser failure, fix that first before trusting any
  speed result

### Never confuse tracks

The feature agent and the image agent are different experiments with different
budgets, observations, and expectations. Speedups on feature PPO do not count as
progress for this browser-image program.
