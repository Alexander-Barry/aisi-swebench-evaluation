# AISI SWE-Bench Evaluation

Making SWE-Bench agent evaluation more informative than binary pass/fail, by extracting richer signal from transcripts, intermediate state, or changes to the evaluation protocol.

## Setup

Requires Docker Desktop, WSL2 with Ubuntu, and Python 3.10+.

The eval pipeline must run under WSL (Linux), because the `swebench` scorer depends on the `resource` module which is not available on Windows.

### Steps

1. Create a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=<your-key>
   INSPECT_EVAL_MODEL=anthropic/claude-sonnet-4-6
   ```

2. Set up a Python venv in WSL and install dependencies:
   ```bash
   python3 -m venv ~/swebench-venv
   source ~/swebench-venv/bin/activate
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

Scripts are numbered in pipeline order:

| Script | Purpose |
|---|---|
| `0_setup.py` | Pull Docker images and download task definitions |
| `1_run_eval.py` | Run 1: baseline eval (5 tasks x 5 epochs, message_limit=30) |
| `2A_summary.py` | Print performance summary from eval logs |
| `2B_viewer.py` | Launch Inspect log viewer for browsing transcripts |
| `2C_flaky_test.py` | Reproduce pre-existing test failure in matplotlib Docker image |
| `3_run_eval_revised.py` | Run 2: increased message limit (150), rate-limit-safe concurrency |

## Tasks

5 instances from SWE-Bench Verified, one per repo:

| Instance | Repo | Difficulty |
|---|---|---|
| `pylint-dev__pylint-4551` | pylint | <15 min fix |
| `psf__requests-1142` | requests | <15 min fix |
| `matplotlib__matplotlib-20859` | matplotlib | <15 min fix |
| `pytest-dev__pytest-5809` | pytest | <15 min fix |
| `scikit-learn__scikit-learn-13779` | scikit-learn | <15 min fix |

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

### Scoring artifact: matplotlib flaky test

`test_warn_big_data_best_loc` fails deterministically (0/20 and 0/100 in isolated runs) in the unmodified Docker image. The test expects a `UserWarning` about large data in `_find_best_position`, but the warning is never emitted in this environment. The agent's patch is correct in all 5 epochs (FAIL_TO_PASS test passes, 87/88 PASS_TO_PASS tests pass), but the pre-existing failure causes SWE-Bench to score 3 epochs as failures.

This is reproducible via `python 2C_flaky_test.py`.

### Run 2 (message_limit=150) -- in progress

Rerunning all tasks with `message_limit=150` and `max_connections=2` (to avoid Anthropic rate limits). The main question is whether pylint can be solved with more steps.

## Technical details

- **Model**: Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`)
- **Agent**: Inspect's default SWE-bench solver (ReAct with bash, python, text_editor tools)
- **Eval config**: 5 epochs per task, temperature=0.5, full API logging
- **Docker images**: Pre-built from [Epoch AI](https://epoch.ai/blog/swebench-docker) (`ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>`)
- **Container layout**: Repo at `/testbed/`, conda env `testbed`
- **Dataset**: `princeton-nlp/SWE-bench_Verified` on HuggingFace
- **Log viewer**: `inspect view --log-dir ./logs` to browse transcripts interactively
