"""
Combined analysis: eval results + LLM judge classifications.

Computes three scoring levels per sample:
  - Raw: the SWE-bench scorer output
  - Flaky-adjusted: corrects for known broken tests in Docker images
  - Judge-adjusted: only requires tests classified REQUIRED or UNCLEAR >= 3/5 runs

Also includes time-to-fix breakdown from 4_analyse.

Usage:
    python 6_analyse.py
"""

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

LOG_DIR = Path("logs")
JUDGE_RESULTS_PATH = Path("results") / "judge_results.json"
JUDGE_THRESHOLD = 3  # out of 5 runs

# Tests known to be broken in the base Docker images (not caused by agent).
KNOWN_BROKEN_TESTS = {
    "lib/matplotlib/tests/test_legend.py::test_warn_big_data_best_loc",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LOG_FILE = LOG_DIR / "2026-03-31T09-49-36-00-00_swe-bench_BByp2eGWdHguVCD5qwHVD2.eval"


def find_log():
    """Return the Run 2 log file."""
    if not LOG_FILE.exists():
        print(f"{LOG_FILE} not found", file=sys.stderr)
        sys.exit(1)
    return LOG_FILE


def load_samples(log_path):
    samples = []
    with zipfile.ZipFile(log_path) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                samples.append(json.loads(z.read(name)))
    return samples


def load_judge_results():
    """Load judge results and compute per-test required status.

    A test is 'required' if classified REQUIRED or UNCLEAR in >= JUDGE_THRESHOLD runs.
    Returns {task_id: {test_name: {"required": bool, "counts": Counter}}}.
    """
    if not JUDGE_RESULTS_PATH.exists():
        print(f"Warning: {JUDGE_RESULTS_PATH} not found, judge-adjusted scores unavailable",
              file=sys.stderr)
        return {}

    with open(JUDGE_RESULTS_PATH) as f:
        data = json.load(f)

    result = {}
    for task_id, task_data in data.items():
        runs = task_data.get("runs", [])
        test_verdicts = {}

        # Collect all test names across runs
        all_tests = set()
        for r in runs:
            all_tests.update(r.get("classifications", {}).keys())

        for test_name in all_tests:
            counts = Counter()
            for r in runs:
                cls = r.get("classifications", {})
                # Try exact match, then substring match
                cat = None
                if test_name in cls:
                    cat = cls[test_name].get("category")
                else:
                    for key, info in cls.items():
                        if test_name in key or key in test_name:
                            cat = info.get("category")
                            break
                if cat:
                    counts[cat] += 1

            required_votes = counts.get("REQUIRED", 0) + counts.get("UNCLEAR", 0)
            test_verdicts[test_name] = {
                "required": required_votes >= JUDGE_THRESHOLD,
                "counts": dict(counts),
            }

        result[task_id] = test_verdicts

    return result


# ---------------------------------------------------------------------------
# Test result parsing
# ---------------------------------------------------------------------------

def parse_test_results(explanation, section):
    """Parse PASS_TO_PASS or FAIL_TO_PASS JSON from the score explanation."""
    if section not in explanation:
        return {}
    rest = explanation[explanation.index(section) + len(section):]
    brace_start = rest.find("{")
    if brace_start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(rest[brace_start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(rest[brace_start:brace_start + i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def match_test_name(test_name, judge_tests):
    """Match a test name from the score explanation to a judge test key."""
    if test_name in judge_tests:
        return test_name
    for key in judge_tests:
        if test_name in key or key in test_name:
            return key
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def raw_score(sample):
    return sample["scores"]["swe_bench_scorer"]["value"]


def flaky_adjusted_score(sample):
    """1.0 if fix is correct, ignoring known broken tests."""
    if raw_score(sample) == 1.0:
        return 1.0

    explanation = sample["scores"]["swe_bench_scorer"].get("explanation", "")
    ftp = parse_test_results(explanation, "FAIL_TO_PASS")
    ptp = parse_test_results(explanation, "PASS_TO_PASS")

    ftp_all_pass = ftp and all(v == "PASSED" for v in ftp.values())
    ptp_failures = {k for k, v in ptp.items() if v != "PASSED"}
    ptp_real_failures = ptp_failures - KNOWN_BROKEN_TESTS

    if ftp_all_pass and not ptp_real_failures:
        return 1.0
    return 0.0


def judge_adjusted_score(sample, judge_results):
    """1.0 if all judge-required FtP tests pass and PtP is clean (minus known broken)."""
    if raw_score(sample) == 1.0:
        return 1.0

    task_id = sample["id"]
    judge_tests = judge_results.get(task_id, {})
    if not judge_tests:
        # No judge data — fall back to flaky-adjusted
        return flaky_adjusted_score(sample)

    explanation = sample["scores"]["swe_bench_scorer"].get("explanation", "")
    ftp = parse_test_results(explanation, "FAIL_TO_PASS")
    ptp = parse_test_results(explanation, "PASS_TO_PASS")

    # Check FtP: only required tests must pass
    for test_name, status in ftp.items():
        key = match_test_name(test_name, judge_tests)
        if key and judge_tests[key]["required"] and status != "PASSED":
            return 0.0

    # Check PtP: ignore known broken tests
    ptp_failures = {k for k, v in ptp.items() if v != "PASSED"}
    ptp_real_failures = ptp_failures - KNOWN_BROKEN_TESTS
    if ptp_real_failures:
        return 0.0

    return 1.0


# ---------------------------------------------------------------------------
# Time-to-fix analysis
# ---------------------------------------------------------------------------

EDIT_COMMANDS = {"str_replace", "create", "insert"}


def find_final_edit_message(sample):
    msgs = sample.get("messages", [])
    last_edit_idx = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls", []):
            if tc.get("function") == "text_editor":
                cmd = tc.get("arguments", {}).get("command", "")
                if cmd in EDIT_COMMANDS:
                    last_edit_idx = i
                    break
    return last_edit_idx


def model_event_for_messages(sample, start, end):
    events = sample.get("events", [])
    model_events = [e for e in events if e.get("event") == "model"]
    msgs = sample.get("messages", [])

    matched = []
    me_idx = 0
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and me_idx < len(model_events):
            if start <= i < end:
                matched.append(model_events[me_idx])
            me_idx += 1
    return matched


def turn_tokens(me):
    usage = me.get("output", {}).get("usage", {})
    in_tok = (usage.get("input_tokens", 0)
              + usage.get("input_tokens_cache_write", 0)
              + usage.get("input_tokens_cache_read", 0))
    out_tok = usage.get("output_tokens", 0)
    return in_tok, out_tok


def phase_metrics(model_events_subset):
    in_tok = sum(turn_tokens(t)[0] for t in model_events_subset)
    out_tok = sum(turn_tokens(t)[1] for t in model_events_subset)
    time_s = sum(t.get("working_time", 0) for t in model_events_subset)
    return in_tok, out_tok, time_s


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def judge_ftp_counts(sample, judge_results):
    """Count pass/total for only judge-required FtP tests."""
    task_id = sample["id"]
    judge_tests = judge_results.get(task_id, {})
    if not judge_tests:
        return None, None

    explanation = sample["scores"]["swe_bench_scorer"].get("explanation", "")
    ftp = parse_test_results(explanation, "FAIL_TO_PASS")

    req_pass = 0
    req_total = 0
    for test_name, status in ftp.items():
        key = match_test_name(test_name, judge_tests)
        if key and judge_tests[key]["required"]:
            req_total += 1
            if status == "PASSED":
                req_pass += 1
    return req_pass, req_total


def print_detail(samples, judge_results):
    print("PER-RUN DETAIL")
    has_judge = bool(judge_results)
    hdr = (f"{'Task':<45} {'Ep':>3} {'Raw':>4} {'Flky':>5}")
    if has_judge:
        hdr += f" {'Jdg':>4}"
    hdr += (f" {'Msgs':>5} {'Lim':>4} {'FtP':>7} {'PtP':>7} {'aPtP':>7}")
    if has_judge:
        hdr += f" {'jFtP':>7}"
    hdr += (f" {'FxMsg':>5} {'FixTok':>8} {'FixT':>5}"
            f" {'VfMsg':>5} {'VfyTok':>8} {'VfyT':>5}")
    print(hdr)
    print("-" * len(hdr))

    for s in sorted(samples, key=lambda x: (x["id"], x["epoch"])):
        raw = "P" if raw_score(s) == 1.0 else "F"
        flaky = "P" if flaky_adjusted_score(s) == 1.0 else "F"
        msgs = s.get("message_count") or len(s.get("messages", []))
        limit = "Y" if s.get("limit") else ""
        time_s = round(s.get("working_time", 0), 1)

        explanation = s["scores"]["swe_bench_scorer"].get("explanation", "")
        ftp = parse_test_results(explanation, "FAIL_TO_PASS")
        ptp = parse_test_results(explanation, "PASS_TO_PASS")
        ftp_pass = sum(1 for v in ftp.values() if v == "PASSED")
        ptp_pass = sum(1 for v in ptp.values() if v == "PASSED")
        ptp_adj_total = sum(1 for k in ptp if k not in KNOWN_BROKEN_TESTS)
        ftp_str = f"{ftp_pass}/{len(ftp)}" if ftp else "?"
        ptp_str = f"{ptp_pass}/{len(ptp)}" if ptp else "?"
        aptp_str = f"{ptp_pass}/{ptp_adj_total}" if ptp else "?"

        usage = s.get("model_usage", {})
        in_tok = sum(u.get("input_tokens", 0) + u.get("input_tokens_cache_write", 0)
                     + u.get("input_tokens_cache_read", 0) for u in usage.values())
        out_tok = sum(u.get("output_tokens", 0) for u in usage.values())

        line = f"{s['id']:<45} {s['epoch']:>3} {raw:>4} {flaky:>5}"
        if has_judge:
            jdg = "P" if judge_adjusted_score(s, judge_results) == 1.0 else "F"
            line += f" {jdg:>4}"
        line += f" {msgs:>5} {limit:>4} {ftp_str:>7} {ptp_str:>7} {aptp_str:>7}"
        if has_judge:
            jp, jt = judge_ftp_counts(s, judge_results)
            jftp_str = f"{jp}/{jt}" if jt is not None else "?"
            line += f" {jftp_str:>7}"
        # Time-to-fix
        all_msgs = s.get("messages", [])
        total_msgs = len(all_msgs)
        last_edit = find_final_edit_message(s)
        if last_edit is None:
            split = total_msgs
        else:
            split = last_edit + 1
            while split < total_msgs and all_msgs[split].get("role") == "tool":
                split += 1
        fix_msgs = split
        vfy_msgs = total_msgs - split
        fix_me = model_event_for_messages(s, 0, split)
        vfy_me = model_event_for_messages(s, split, total_msgs)
        fi, fo, ft = phase_metrics(fix_me)
        vi, vo, vt = phase_metrics(vfy_me)

        line += (f" {fix_msgs:>5} {fi+fo:>8,} {ft:>4.0f}s"
                 f" {vfy_msgs:>5} {vi+vo:>8,} {vt:>4.0f}s")
        print(line)

    # Totals
    n = len(samples)
    raw_p = sum(1 for s in samples if raw_score(s) == 1.0)
    flaky_p = sum(1 for s in samples if flaky_adjusted_score(s) == 1.0)
    judge_p = sum(1 for s in samples if judge_adjusted_score(s, judge_results) == 1.0) if has_judge else flaky_p
    line = f"{'TOTAL':<45}     {raw_p}/{n:<4} {flaky_p}/{n:<4}"
    if has_judge:
        line += f" {judge_p}/{n:<3}"
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = find_log()
    print(f"Log: {log_path.name}\n")

    samples = load_samples(log_path)
    samples.sort(key=lambda x: (x["id"], x["epoch"]))

    judge_results = load_judge_results()

    print_detail(samples, judge_results)


if __name__ == "__main__":
    main()
