"""
Run SWE-Bench Verified evaluation on 5 selected tasks using Inspect.

Uses the default Inspect SWE-bench agent (ReAct with bash tool),
Epoch AI's pre-built Docker images, and Claude Sonnet 4.6.

Runs 5 epochs (attempts) per task with temperature=0.5 for variation.
Full transcripts and raw API calls are logged.

Usage:
    python 1_run_eval.py
"""

from dotenv import load_dotenv

load_dotenv()

from inspect_ai import eval
from inspect_evals.swe_bench import swe_bench

TASK_IDS = [
    "pylint-dev__pylint-4551",
    "psf__requests-1142",
    "matplotlib__matplotlib-20859",
    "pytest-dev__pytest-5809",
    "scikit-learn__scikit-learn-13779",
]

task = swe_bench()

logs = eval(
    task,
    model="anthropic/claude-sonnet-4-6",
    sample_id=TASK_IDS,
    epochs=5,
    temperature=0.5,
    log_dir="./logs",
    log_model_api=True,
    message_limit=30,
)

print(f"Evaluation complete. {len(logs)} log(s) written to ./logs/")
