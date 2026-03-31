"""
Print a summary of evaluation performance for each task.

For each task x epoch: pass/fail, message count, token usage.
Reads from the Inspect eval log file.

Usage:
    python 2A_summary.py
"""

import json
import sys
import zipfile
from pathlib import Path

LOG_DIR = Path("logs")


def find_log():
    """Find the most recent .eval log file."""
    evals = sorted(LOG_DIR.glob("*.eval"))
    if not evals:
        print("No .eval files found in logs/", file=sys.stderr)
        sys.exit(1)
    return evals[-1]


def load_samples(log_path):
    """Load all samples from individual files (summaries.json truncates long fields)."""
    samples = []
    with zipfile.ZipFile(log_path) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                s = json.loads(z.read(name))
                samples.append(s)
    return samples


def parse_test_counts(explanation, section):
    """Count passed/total tests in a PASS_TO_PASS or FAIL_TO_PASS section."""
    if section not in explanation:
        return 0, 0
    idx = explanation.index(section)
    rest = explanation[idx + len(section):]
    brace_start = rest.find("{")
    if brace_start == -1:
        return 0, 0
    depth = 0
    for i, ch in enumerate(rest[brace_start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    results = json.loads(rest[brace_start:brace_start + i + 1])
                    passed = sum(1 for v in results.values() if v == "PASSED")
                    return passed, len(results)
                except json.JSONDecodeError:
                    return 0, 0
    return 0, 0


def main():
    log_path = find_log()
    print(f"Log: {log_path.name}\n")

    samples = load_samples(log_path)

    # Group by task
    tasks = {}
    for s in samples:
        tasks.setdefault(s["id"], []).append(s)

    # Print per-task summary
    print(f"{'Task':<45} {'Pass':>5} {'Fail':>5} {'Rate':>6}")
    print("=" * 65)
    for task_id in sorted(tasks):
        runs = tasks[task_id]
        n_pass = sum(1 for r in runs if r["scores"]["swe_bench_scorer"]["value"] == 1.0)
        n_fail = len(runs) - n_pass
        rate = n_pass / len(runs)
        print(f"{task_id:<45} {n_pass:>5} {n_fail:>5} {rate:>5.0%}")

    # Detailed per-run table
    print(f"\n{'Task':<45} {'Ep':>3} {'Score':>6} {'Msgs':>5} {'Limit':>6} "
          f"{'FtP':>7} {'PtP':>7} "
          f"{'InTok':>8} {'OutTok':>8} {'TotalTok':>9} {'Time(s)':>8}")
    print("-" * 125)

    for s in sorted(samples, key=lambda x: (x["id"], x["epoch"])):
        score = s["scores"]["swe_bench_scorer"]["value"]
        result = "PASS" if score == 1.0 else "FAIL"
        msgs = s.get("message_count") or len(s.get("messages", []))
        limit = "YES" if s.get("limit") else ""
        time_s = round(s.get("working_time", 0), 1)

        # Test results
        explanation = s["scores"]["swe_bench_scorer"].get("explanation", "")
        ftp_pass, ftp_total = parse_test_counts(explanation, "FAIL_TO_PASS")
        ptp_pass, ptp_total = parse_test_counts(explanation, "PASS_TO_PASS")
        ftp_str = f"{ftp_pass}/{ftp_total}" if ftp_total else "?"
        ptp_str = f"{ptp_pass}/{ptp_total}" if ptp_total else "?"

        # Token usage (sum across all models, though typically just one)
        usage = s.get("model_usage", {})
        input_tok = 0
        output_tok = 0
        total_tok = 0
        for model, u in usage.items():
            input_tok += u.get("input_tokens", 0) + u.get("input_tokens_cache_write", 0) + u.get("input_tokens_cache_read", 0)
            output_tok += u.get("output_tokens", 0)
            total_tok += u.get("total_tokens", 0)

        print(f"{s['id']:<45} {s['epoch']:>3} {result:>6} {msgs:>5} {limit:>6} "
              f"{ftp_str:>7} {ptp_str:>7} "
              f"{input_tok:>8,} {output_tok:>8,} {total_tok:>9,} {time_s:>8}")


if __name__ == "__main__":
    main()
