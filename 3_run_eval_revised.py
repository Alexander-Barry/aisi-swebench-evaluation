"""
Second evaluation run: increased message limit, rate-limit-safe concurrency.

Changes from 1_run_eval.py:
  - max_connections=2 to stay under Anthropic rate limits (30K tok/min, 50 req/min)
  - message_limit raised from 30 to 150

Usage:
    python 3_run_eval_revised.py
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
    message_limit=150,
    max_connections=2,
)

print(f"Evaluation complete. {len(logs)} log(s) written to ./logs/")
