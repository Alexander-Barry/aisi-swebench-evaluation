"""
Setup script for AISI SWE-Bench evaluation.

Downloads the 5 selected SWE-Bench Verified task definitions and pulls
the corresponding pre-built Docker images from Epoch AI's registry.

Usage:
    python setup.py
"""

import json
import subprocess
import sys

TASK_IDS = [
    "pylint-dev__pylint-4551",
    "psf__requests-1142",
    "matplotlib__matplotlib-20859",
    "pytest-dev__pytest-5809",
    "scikit-learn__scikit-learn-13779",
]

IMAGE_REGISTRY = "ghcr.io/epoch-research/swe-bench.eval.x86_64"


def pull_images():
    """Pull all task Docker images from Epoch's registry."""
    for task_id in TASK_IDS:
        image = f"{IMAGE_REGISTRY}.{task_id}:latest"
        print(f"Pulling {image} ...")
        result = subprocess.run(["docker", "pull", image], capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: Failed to pull {image}", file=sys.stderr)
            sys.exit(1)
        print()


def download_tasks():
    """Download task definitions from the SWE-Bench Verified dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    selected = [dict(row) for row in ds if row["instance_id"] in TASK_IDS]

    if len(selected) != len(TASK_IDS):
        found = {t["instance_id"] for t in selected}
        missing = set(TASK_IDS) - found
        print(f"WARNING: missing tasks: {missing}", file=sys.stderr)

    with open("tasks.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved {len(selected)} tasks to tasks.json")


if __name__ == "__main__":
    print("=== Downloading task definitions ===")
    download_tasks()
    print()
    print("=== Pulling Docker images ===")
    pull_images()
    print()
    print("Done. All 5 task environments are ready.")
