"""
Analyse evaluation results: performance summary and time-to-fix breakdown.

Prints:
  1. Per-task pass rates (with adjusted scores correcting for known flaky tests)
  2. Per-run detail: messages, tokens, test results
  3. Time-to-fix analysis: separates "writing the fix" from "verifying after"

Usage:
    python 4_analyse.py
"""

import json
import sys
import zipfile
from pathlib import Path

LOG_DIR = Path("logs")

# Tests known to be broken in the base Docker images (not caused by agent).
# Identified by running tests on unmodified containers.
KNOWN_BROKEN_TESTS = {
    "lib/matplotlib/tests/test_legend.py::test_warn_big_data_best_loc",
}


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


def adjusted_score(sample):
    """Return 1.0 if the agent's fix is correct, ignoring known broken tests."""
    score_data = sample["scores"]["swe_bench_scorer"]
    if score_data["value"] == 1.0:
        return 1.0

    explanation = score_data.get("explanation", "")
    ftp = parse_test_results(explanation, "FAIL_TO_PASS")
    ptp = parse_test_results(explanation, "PASS_TO_PASS")

    ftp_all_pass = ftp and all(v == "PASSED" for v in ftp.values())
    ptp_failures = {k for k, v in ptp.items() if v != "PASSED"}
    ptp_real_failures = ptp_failures - KNOWN_BROKEN_TESTS

    if ftp_all_pass and not ptp_real_failures:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Time-to-fix analysis
# ---------------------------------------------------------------------------

EDIT_COMMANDS = {"str_replace", "create", "insert"}


def find_final_edit_message(sample):
    """Find the message index of the last text_editor write (str_replace/create/insert).

    Returns the index into sample["messages"], or None if no edits found.
    """
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
    """Find model events that correspond to messages in [start, end).

    Model events map 1:1 with assistant messages. We walk the messages
    and match each assistant message to the next model event in order.
    """
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
    """Return (input_tokens, output_tokens) for a model event."""
    usage = me.get("output", {}).get("usage", {})
    in_tok = (usage.get("input_tokens", 0)
              + usage.get("input_tokens_cache_write", 0)
              + usage.get("input_tokens_cache_read", 0))
    out_tok = usage.get("output_tokens", 0)
    return in_tok, out_tok


def phase_metrics(model_events_subset):
    """Compute aggregate token and time metrics for a set of model events."""
    in_tok = sum(turn_tokens(t)[0] for t in model_events_subset)
    out_tok = sum(turn_tokens(t)[1] for t in model_events_subset)
    time_s = sum(t.get("working_time", 0) for t in model_events_subset)
    return in_tok, out_tok, time_s


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(samples):
    print("=" * 80)
    print("TASK SUMMARY")
    print("=" * 80)

    tasks = {}
    for s in samples:
        tasks.setdefault(s["id"], []).append(s)

    print(f"\n{'Task':<45} {'Raw':>7} {'Adj':>7}")
    print("-" * 62)
    for task_id in sorted(tasks):
        runs = tasks[task_id]
        n = len(runs)
        raw = sum(1 for r in runs if r["scores"]["swe_bench_scorer"]["value"] == 1.0)
        adj = sum(1 for r in runs if adjusted_score(r) == 1.0)
        flag = " *" if adj != raw else ""
        print(f"{task_id:<45} {raw}/{n:>3} {adj}/{n:>3}{flag}")
    print("\n* = adjusted differs (known broken test excluded)\n")


def print_detail(samples):
    print("=" * 80)
    print("PER-RUN DETAIL")
    print("=" * 80)

    hdr = (f"{'Task':<45} {'Ep':>3} {'Score':>5} {'Adj':>4} {'Msgs':>5} "
           f"{'Lim':>4} {'FtP':>7} {'PtP':>7} "
           f"{'InTok':>8} {'OutTok':>8} {'Time':>7}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for s in sorted(samples, key=lambda x: (x["id"], x["epoch"])):
        raw = "P" if s["scores"]["swe_bench_scorer"]["value"] == 1.0 else "F"
        adj = "P" if adjusted_score(s) == 1.0 else "F"
        msgs = s.get("message_count") or len(s.get("messages", []))
        limit = "Y" if s.get("limit") else ""
        time_s = round(s.get("working_time", 0), 1)

        explanation = s["scores"]["swe_bench_scorer"].get("explanation", "")
        ftp = parse_test_results(explanation, "FAIL_TO_PASS")
        ptp = parse_test_results(explanation, "PASS_TO_PASS")
        ftp_pass = sum(1 for v in ftp.values() if v == "PASSED")
        ptp_pass = sum(1 for v in ptp.values() if v == "PASSED")
        ftp_str = f"{ftp_pass}/{len(ftp)}" if ftp else "?"
        ptp_str = f"{ptp_pass}/{len(ptp)}" if ptp else "?"

        usage = s.get("model_usage", {})
        in_tok = sum(u.get("input_tokens", 0) + u.get("input_tokens_cache_write", 0)
                     + u.get("input_tokens_cache_read", 0) for u in usage.values())
        out_tok = sum(u.get("output_tokens", 0) for u in usage.values())

        print(f"{s['id']:<45} {s['epoch']:>3} {raw:>5} {adj:>4} {msgs:>5} "
              f"{limit:>4} {ftp_str:>7} {ptp_str:>7} "
              f"{in_tok:>8,} {out_tok:>8,} {time_s:>7}")
    print()


def print_time_to_fix(samples):
    print("=" * 80)
    print("TIME-TO-FIX ANALYSIS")
    print("Last text_editor edit = final code change; everything after = verification")
    print("Msg counts match the Msgs column in the detail table")
    print("=" * 80)

    hdr = (f"{'Task':<45} {'Ep':>3} {'Adj':>4} {'FtP':>7} {'PtP':>7}  "
           f"{'FxMsg':>5} {'FixIn':>8} {'FixOut':>7} {'FixT':>6}  "
           f"{'VfMsg':>5} {'VfyIn':>8} {'VfyOut':>7} {'VfyT':>6}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for s in sorted(samples, key=lambda x: (x["id"], x["epoch"])):
        adj = "P" if adjusted_score(s) == 1.0 else "F"

        # Adjusted test counts (exclude known broken tests from PtP failures)
        explanation = s["scores"]["swe_bench_scorer"].get("explanation", "")
        ftp = parse_test_results(explanation, "FAIL_TO_PASS")
        ptp = parse_test_results(explanation, "PASS_TO_PASS")
        ftp_pass = sum(1 for v in ftp.values() if v == "PASSED")
        ptp_pass = sum(1 for v in ptp.values() if v == "PASSED")
        ptp_adj_total = sum(1 for k in ptp if k not in KNOWN_BROKEN_TESTS)
        ftp_str = f"{ftp_pass}/{len(ftp)}" if ftp else "?"
        ptp_str = f"{ptp_pass}/{ptp_adj_total}" if ptp else "?"

        msgs = s.get("messages", [])
        total_msgs = len(msgs)
        last_edit = find_final_edit_message(s)

        if last_edit is None:
            split = total_msgs
        else:
            split = last_edit + 1
            while split < total_msgs and msgs[split].get("role") == "tool":
                split += 1

        fix_msgs = split
        vfy_msgs = total_msgs - split

        fix_me = model_event_for_messages(s, 0, split)
        vfy_me = model_event_for_messages(s, split, total_msgs)

        fi, fo, ft = phase_metrics(fix_me)
        vi, vo, vt = phase_metrics(vfy_me)

        print(f"{s['id']:<45} {s['epoch']:>3} {adj:>4} {ftp_str:>7} {ptp_str:>7}  "
              f"{fix_msgs:>5} {fi:>8,} {fo:>7,} {ft:>5.0f}s  "
              f"{vfy_msgs:>5} {vi:>8,} {vo:>7,} {vt:>5.0f}s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = find_log()
    print(f"Log: {log_path.name}\n")

    samples = load_samples(log_path)
    samples.sort(key=lambda x: (x["id"], x["epoch"]))

    print_summary(samples)
    print_detail(samples)
    print_time_to_fix(samples)


if __name__ == "__main__":
    main()
