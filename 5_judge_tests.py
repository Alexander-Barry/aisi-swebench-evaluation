"""
LLM judge: are FAIL_TO_PASS tests actually required to address the stated issue?

For each task, extracts the problem statement and FAIL_TO_PASS test code,
then asks Claude Opus 4.6 to classify each test as:
  - REQUIRED: directly tests the stated issue
  - IMPLEMENTATION_SPECIFIC: useful but tests a specific implementation, not required
  - UNRELATED: no clear connection to the issue
  - UNCLEAR: ambiguous relationship

Usage:
    python 5_judge_tests.py
"""

import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

IMAGE_REGISTRY = "ghcr.io/epoch-research/swe-bench.eval.x86_64"
JUDGE_MODEL = "claude-opus-4-6"


def load_tasks():
    with open("tasks.json") as f:
        return json.load(f)


def extract_test_code(task):
    """Apply the test patch in a Docker container and read the test file."""
    instance_id = task["instance_id"]
    image = f"{IMAGE_REGISTRY}.{instance_id}:latest"
    test_patch = task["test_patch"]
    fail_to_pass = json.loads(task["FAIL_TO_PASS"])

    # Get unique test file paths from test names
    test_files = set()
    for test_name in fail_to_pass:
        file_path = test_name.split("::")[0]
        test_files.add(file_path)

    # Apply test patch and read the test files
    # We pipe the patch via stdin and then cat the files
    script = "cd /testbed && git apply --allow-empty - <<'PATCH_EOF'\n"
    script += test_patch
    script += "\nPATCH_EOF\n"
    for tf in sorted(test_files):
        script += f"echo '=== FILE: {tf} ==='\n"
        script += f"cat {tf}\n"

    result = subprocess.run(
        ["docker", "run", "--rm", "-i", image, "bash", "-c", script],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        print(f"  Warning: docker failed for {instance_id}: {result.stderr[:200]}",
              file=sys.stderr)

    return result.stdout


def build_prompt(task, test_code):
    """Build the judge prompt for a single task."""
    problem_statement = task["problem_statement"]
    fail_to_pass = json.loads(task["FAIL_TO_PASS"])

    test_list = "\n".join(f"  - {t}" for t in fail_to_pass)

    return f"""You are evaluating whether automated test cases are strictly necessary to verify a fix for a specific software issue.

## Context
A benchmark (SWE-Bench) evaluates AI agents' ability to fix software issues. For each issue, there are "FAIL_TO_PASS" tests that must all pass for a fix to be scored as correct.

The question is: are all these tests actually required to address the stated issue, or do some test implementation-specific details that go beyond what the issue asks for?

## Issue
{problem_statement}

## FAIL_TO_PASS Tests
The following tests must ALL pass for a fix to be scored as correct:
{test_list}

## Test Code
{test_code}

## Instructions
For each FAIL_TO_PASS test listed above, classify it as one of:

- REQUIRED: This test is strictly necessary to verify the stated issue is fixed. There is no reasonable alternative implementation that would address the issue without passing this test. Note that a test which relates to the issue but tests a specific API shape, output format, or design choice not mentioned in the issue is not required — it is implementation-specific.
- IMPLEMENTATION_SPECIFIC: This test is useful or tests a feature of a specific implementation, but passing it is not strictly required to address the stated issue. An alternative valid fix might not pass this test.
- UNRELATED: This test has no clear connection to the stated issue.
- UNCLEAR: There is insufficient information to determine if this test is required or not.

Return ONLY a JSON object (no markdown fences) with this structure:
{{
  "classifications": {{
    "<test_name>": {{
      "category": "REQUIRED | IMPLEMENTATION_SPECIFIC | UNRELATED | UNCLEAR",
      "reasoning": "brief explanation"
    }}
  }}
}}"""


def judge_task(client, task, test_code):
    """Send a task to the judge model and parse the response."""
    prompt = build_prompt(task, test_code)

    text = ""
    with client.messages.stream(
        model=JUDGE_MODEL,
        max_tokens=32000,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk

    text = text.strip()

    # Try to parse JSON (handle potential markdown fences)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  Warning: could not parse judge response as JSON", file=sys.stderr)
        print(f"  Raw response: {text[:500]}", file=sys.stderr)
        return {"classifications": {}, "raw": text}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="LLM judge for FAIL_TO_PASS test relevance")
    parser.add_argument("filter", nargs="?", help="Filter to tasks matching this string")
    parser.add_argument("--runs", type=int, default=5, help="Number of judge runs (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks()
    client = anthropic.Anthropic()

    if args.filter:
        tasks = [t for t in tasks if args.filter in t["instance_id"]]
        if not tasks:
            print(f"No task matching '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    n_runs = args.runs
    out_path = Path("results") / "judge_results.json"
    out_path.parent.mkdir(exist_ok=True)

    # Load existing results if any
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Extract test code once per task (doesn't change between runs)
    task_test_code = {}
    for task in tasks:
        instance_id = task["instance_id"]
        print(f"\nExtracting test code for {instance_id}...")
        task_test_code[instance_id] = extract_test_code(task)

    # Run judge n_runs times
    for run in range(1, n_runs + 1):
        print(f"\n{'#' * 80}")
        print(f"RUN {run}/{n_runs}")
        print(f"{'#' * 80}")

        for task in tasks:
            instance_id = task["instance_id"]
            fail_to_pass = json.loads(task["FAIL_TO_PASS"])

            print(f"\n  Task: {instance_id} ({len(fail_to_pass)} tests)")
            print(f"  Sending to {JUDGE_MODEL}...")

            result = judge_task(client, task, task_test_code[instance_id])

            # Store per-run result
            all_results.setdefault(instance_id, {"runs": []})
            if "runs" not in all_results[instance_id]:
                # Migrate from old single-result format
                all_results[instance_id] = {"runs": []}
            all_results[instance_id]["runs"].append(result)

            # Print this run's classifications
            classifications = result.get("classifications", {})
            for test_name, info in classifications.items():
                cat = info.get("category", "?")
                short_name = test_name.split("::")[-1] if "::" in test_name else test_name
                print(f"    {cat:<25s} {short_name}")

        # Save after each run (so partial results survive interruption)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Aggregate summary
    print(f"\n{'=' * 80}")
    print(f"AGGREGATE SUMMARY ({n_runs} runs)")
    print(f"{'=' * 80}\n")

    for task in tasks:
        instance_id = task["instance_id"]
        runs = all_results.get(instance_id, {}).get("runs", [])
        fail_to_pass = json.loads(task["FAIL_TO_PASS"])

        print(f"  {instance_id}:")

        # Count classifications per test across runs
        for test_name in fail_to_pass:
            counts = {}
            for r in runs:
                cls = r.get("classifications", {})
                # Match test name (judge may use full or short name)
                cat = None
                for key, info in cls.items():
                    if test_name in key or key in test_name:
                        cat = info.get("category", "?")
                        break
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1

            short_name = test_name.split("::")[-1] if "::" in test_name else test_name
            counts_str = ", ".join(f"{v}x {k}" for k, v in sorted(counts.items()))
            print(f"    {short_name}: {counts_str}")
        print()

    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
