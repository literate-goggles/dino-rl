# autoresearch

This experiment asks the LLM to **maximize the evaluation score** of the
**browser-image PPO** agent for Chrome Dino, pushing the deterministic eval
score toward a target of **10,000**.

## Context

The repo has two distinct PPO tracks:

| Track | Backend | Observation | Status |
|---|---|---|---|
| Feature PPO | Python simulator | 10 engineered features | Strong; eval avg ~1400-2140 |
| Image PPO | Real Chrome + ChromeDriver | 4 x 84 x 84 grayscale stack | Current research target |

This file is about the **image PPO** track only.

Current browser-image setup:

- Environment backend: real `chrome://dino` through `dino_rl/browser_env.py`
- Observation: 4 stacked grayscale frames of shape `(4, 84, 84)`
- Preprocessing: crop to gameplay strip, mask score HUD, max-pool recent frames
- Action repeat: `2`
- Action space: `3` actions
  - `0 = noop`
  - `1 = jump`
  - `2 = duck / hold down`
- Reward:
  - positive reward from scaled `distanceRan` progress
  - `-1.0` on crash
- Parallelism: a threaded vectorized env (`--num-envs`) runs many headless
  Chrome workers feeding one learner. This already landed and gives ~5x more
  env steps per hour; use it so each experiment trains on far more data.
- Browser recovery:
  - the env is expected to recover from dead sessions / tab crashes instead of
    killing the trainer (recovery is per-worker and isolated)

## Goal

The objective of this program is to **make the agent learn to play better** —
to raise the deterministic evaluation score as high as possible, with the
explicit target of reaching an **eval avg of 10,000**. This is a
sample-efficiency and learning-quality problem, not a throughput problem.

- **primary target: maximize `eval/avg_score`** (deterministic greedy eval)
- secondary target: raise `eval/max_score` and keep eval improving over the run
- secondary target: stable, monotonic learning — eval should trend up, not
  spike and collapse

Treat training throughput as a **means, not the objective**. The parallel-worker
work is done; faster training matters only because it lets the agent see more
data and learn more within the budget. Within that, always prefer the change
that **learns better**, even if it is somewhat slower per step.

The current image agent evals around ~75-150. Reaching 10,000 is a large jump
(the feature track gets there); expect it to require many hours of accumulated
training, better learning dynamics, and likely several kept changes. Resume from
the best checkpoint so progress compounds across the campaign.

## Setup

To start a new research pass, work with the user to:

1. Propose a run tag based on the date, for example `jun24-image`
2. Create a fresh branch `autoresearch/<tag>` from current `master`
3. Read the in-scope files:
   - `dino_rl/algorithms/ppo.py`
   - `dino_rl/browser_env.py`
   - `dino_rl/play_browser.py`
   - `dino_rl/policy_loader.py`
4. Confirm that Chrome/ChromeDriver browser control is functional
5. Establish a **baseline eval score** for the current code before changing
   anything
6. Kick off the experiment loop

## Experimentation

Each experiment runs for a **fixed wall-clock budget of 1 hour** on a single
GPU, so variants are compared on equal footing. Because reaching 10,000 needs
sustained training, **resume from the best checkpoint** so eval progress
accumulates across experiments toward the target.

**Always run on GPU 2.** Pin every training, eval, and profiling run to that
device with `CUDA_VISIBLE_DEVICES=2` (as in the commands below). Do not use any
other GPU — the other devices may be in use by other work.

Launch command from repo root (parallel workers on; resume from best):

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m dino_rl.algorithms.ppo \
  --env-backend browser \
  --observation-mode image \
  --num-envs 16 \
  --time-budget-sec 3600 \
  --eval-every 5 \
  --print-every 1 \
  --init-checkpoint checkpoints/dino_ppo_browser_image_best.pth \
  > run_browser_image.log 2>&1
```

For a cold start (no usable best checkpoint), drop `--init-checkpoint`.

### What you CAN modify

- `dino_rl/algorithms/ppo.py`
- `dino_rl/browser_env.py`
- `dino_rl/play_browser.py`
- `dino_rl/policy_loader.py`

Anything that improves how well the agent learns is fair game. Levers worth
exploring for **eval quality** include:

- reward shaping: `distance_reward_scale`, the crash penalty, `score_delta`
  shaping, reward clipping / normalization
- exploration: entropy coefficient and its schedule, action-repeat tuning
- optimization: learning rate and schedule, clip epsilon, PPO epochs, minibatch
  size, value-loss coefficient, GAE `gamma`/`lambda`, advantage normalization,
  grad clipping
- network: CNN encoder capacity/architecture, latent size, separate vs shared
  actor-critic trunk, frame stack
- batch size / data: number of parallel envs and rollout length (bigger
  on-policy batches already helped eval), how much total data per update
- observation pipeline quality: crop/mask/resize choices that make the obstacle
  signal cleaner (not just cheaper)

### What you CANNOT modify

- `dino_rl/env.py` for the purpose of making the browser-image task easier
- install new packages
- do not "improve" eval by gaming it — e.g. evaluating with acceleration off
  when it should be on, shortening eval episodes, hand-tuning to the eval seed,
  or otherwise inflating the reported score without real skill gains
- silently switch the experiment back to simulator features

The task here is specifically to make PPO on **browser images** actually play
better and drive eval score toward 10,000.

## Key Technical Facts

- Action space is `3`, so max entropy is `ln(3) ~= 1.099`
- `TARGET_EVAL_SCORE = 10000`; the trainer stops early if eval reaches it
- Current image defaults:
  - `rollout_len = 512` (per env)
  - `num_envs = 1` by default; pass `--num-envs N` for parallel workers
  - `minibatch_size = 64`
  - `action_repeat = 2`
  - `eval_every = 5`, `eval_episodes = 7` (deterministic greedy eval)
  - `score_delta_coeff = 0.0`
- This image path is plain PPO with a CNN encoder (no auxiliary losses)
- Parallel workers already gave a large speedup; in a 1-hour run at `--num-envs
  16` the agent collected ~120k steps and eval rose from ~75 to ~130. More data
  and larger on-policy batches help — but closing the gap to 10,000 will need
  genuinely better learning, not just more steps.
- Browser failures such as `tab crashed`, disconnected sessions, or dead
  ChromeDriver connections are recoverable env failures, not reasons for the
  trainer to exit, and recovery should stay fast.

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

Key metrics to watch (eval first):

- **`eval/avg_score`** — the target metric; drive it toward 10,000
- `eval/max_score` and the spread between eval episodes — consistency matters
- `train/entropy` — near `1.099` means the policy went random (no learning);
  near `0` too early means premature collapse / lost exploration
- `train/approx_kl`, `train/clip_fraction` — update stability
- `train/value_loss` — whether the critic is tracking returns
- browser recovery messages — slow or frequent recovery wastes training time

## Logging Results

When an experiment finishes, log it to `results.tsv` as tab-separated text:

```text
commit	eval_avg	eval_max	total_steps	status	description
```

Columns:

1. short git commit hash
2. `eval_avg` (deterministic eval average — the primary metric)
3. `eval_max` (best single eval episode)
4. `total_steps` completed within the 1-hour budget (context)
5. status: `keep`, `discard`, or `crash`
6. short description of the change attempted

Do not commit `results.tsv`.

## The Experiment Loop

Run on a dedicated branch such as `autoresearch/jun24-image`.

Loop:

1. Inspect the current branch and commit
2. Make one focused change aimed at **better learning / higher eval**
3. Commit the change
4. Run a 1-hour browser-image PPO experiment (resume from the best checkpoint)
5. Read the results from the log: eval avg trajectory and final/best eval
6. If the run crashed:
   - inspect the traceback
   - distinguish between a browser-recovery bug, a trainer bug, and a bad idea
7. Record the result in `results.tsv`
8. If `eval_avg` improved **and** learning stayed stable, keep the commit
9. If `eval_avg` did not improve, or learning destabilized, revert the change
   and continue from the last good point

### Diagnose learning first

Before guessing, look at the eval trajectory and the training signals (entropy,
KL, clip fraction, value loss, advantage stats). Identify what is actually
limiting the agent — exploration collapse, an unstable update, a weak critic, a
reward that does not reward survival — and attack that. Prefer changes justified
by a measured failure mode over speculative tweaks.

### Timeout rule

- Budget per experiment: `3600` seconds
- If a run goes materially past `1 hour` without the trainer honoring the time
  budget, stop it manually and treat that as a failure in the experiment loop

### Crash rule

- A recoverable Chrome crash should not kill PPO
- If PPO still exits on browser failure, fix that first before trusting any
  eval result

### Never confuse tracks

The feature agent and the image agent are different experiments with different
backends, observations, and expectations. Eval gains on feature PPO do not count
as progress for this browser-image program.
