# AISI SWE-Bench Evaluation

Making SWE-Bench agent evaluation more informative than binary pass/fail, by extracting richer signal from transcripts, intermediate state, or changes to the evaluation protocol.

## Setup

Requires Docker Desktop, WSL2 with Ubuntu, and Python 3.10+. Docker Desktop's WSL integration must be enabled for the Ubuntu distro (Settings > Resources > WSL Integration).

The eval pipeline runs under WSL (Linux) because the `swebench` scorer depends on Python's `resource` module, which is Unix-only. All scripts should be run from WSL, not Windows directly.

### Steps

1. Create a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=<your-key>
   INSPECT_EVAL_MODEL=anthropic/claude-sonnet-4-6
   ```

2. Set up a Python venv **inside WSL** (not on the Windows-mounted filesystem, which causes venv issues):
   ```bash
   python3 -m venv ~/swebench-venv
   source ~/swebench-venv/bin/activate
   cd /mnt/c/Users/<you>/Documents/'AISI Work Trial'
   pip install -r requirements.txt
   ```

3. Patch a known version incompatibility and pull Docker images:
   ```bash
   python3 patch_timeout.py
   python3 0_setup.py
   ```

### Known issue

As of `inspect-ai==0.3.201` / `inspect-evals==0.6.0`, the SWE-bench eval sets `bash_session(timeout=180)` but `inspect_ai` requires a minimum of 210s. `patch_timeout.py` fixes this automatically.

## Pipeline

Scripts are numbered in execution order:

| Script | Purpose |
|---|---|
| `0_setup.py` | Pull Docker images and download task definitions |
| `1_run_eval.py` | Run 1: baseline eval (5 tasks x 5 epochs, message_limit=30) |
| `2A_summary.py` | Print performance summary from eval logs |
| `2B_viewer.py` | Launch Inspect log viewer for browsing transcripts |
| `2C_flaky_test.py` | Reproduce pre-existing test failure in matplotlib Docker image |
| `3_run_eval_revised.py` | Run 2: increased message limit (150), rate-limit-safe concurrency |
| `4_analyse.py` | Performance summary with flaky-test adjustment and time-to-fix breakdown |
| `5_judge_tests.py` | LLM judge: classify FAIL_TO_PASS tests as required vs implementation-specific |
| `6_analyse.py` | Combined analysis: raw, flaky-adjusted, and judge-adjusted scoring |

## Tasks

5 instances from SWE-Bench Verified, one per repo:

| Instance | Repo |
|---|---|
| `pylint-dev__pylint-4551` | pylint |
| `psf__requests-1142` | requests |
| `matplotlib__matplotlib-20859` | matplotlib |
| `pytest-dev__pytest-5809` | pytest |
| `scikit-learn__scikit-learn-13779` | scikit-learn |

## Results

### Run 1 (message_limit=30)

| Task | Raw | Adjusted | Notes |
|---|---|---|---|
| `psf__requests-1142` | 5/5 | 5/5 | Consistent solve |
| `pytest-dev__pytest-5809` | 5/5 | 5/5 | Fast, often 15 messages |
| `scikit-learn__scikit-learn-13779` | 5/5 | 5/5 | Consistent solve |
| `matplotlib__matplotlib-20859` | 2/5 | **5/5** | Correct patch all 5 epochs; failures caused by pre-existing broken test in Docker image (see below) |
| `pylint-dev__pylint-4551` | 0/5 | 0/5 | All epochs hit 30-message limit |

**Overall: 68% raw, 80% adjusted (correcting for flaky test)**

### Run 2 (message_limit=150)

Rerun with `message_limit=150` and `max_connections=2` (to stay under Anthropic rate limits). Run `python 6_analyse.py` to see full results including judge-adjusted scores.

### Scoring artifact: matplotlib flaky test

`test_warn_big_data_best_loc` fails deterministically (0/20 and 0/100 in isolated runs) in the unmodified Docker image. The test expects a `UserWarning` about large data in `_find_best_position`, but the warning is never emitted in this environment. The agent's patch is correct in all 5 epochs (FAIL_TO_PASS test passes, 87/88 PASS_TO_PASS tests pass), but the pre-existing failure causes SWE-Bench to score 3 epochs as failures.

Reproducible via `python 2C_flaky_test.py`.

### LLM judge for test relevance

`5_judge_tests.py` uses Claude Opus to classify each FAIL_TO_PASS test as REQUIRED, IMPLEMENTATION_SPECIFIC, UNRELATED, or UNCLEAR. This provides a second axis of scoring adjustment beyond flaky-test correction: a fix that passes all *required* tests but fails an *implementation-specific* test may still be a valid solution.

Results are aggregated over 5 judge runs (majority vote at threshold 3/5). See `6_analyse.py` for the combined scoring table.

## Technical details

- **Model**: Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`)
- **Agent**: Inspect's default SWE-bench solver (ReAct with bash, python, text_editor tools)
- **Eval config**: 5 epochs per task, temperature=0.5, full API logging
- **Docker images**: Pre-built from [Epoch AI](https://epoch.ai/blog/swebench-docker) (`ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>`)
- **Container layout**: Repo at `/testbed/`, conda env `testbed`
- **Dataset**: `princeton-nlp/SWE-bench_Verified` on HuggingFace
- **Log viewer**: `inspect view --log-dir ./logs` to browse transcripts interactively
