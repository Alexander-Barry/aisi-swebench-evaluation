"""
Test whether test_warn_big_data_best_loc is broken in the base Docker image.

Runs the test 20 times in a fresh matplotlib container (no agent patch applied)
to confirm the failure is pre-existing, not caused by the agent.

Usage:
    python 2C_flaky_test.py
"""

import subprocess
import sys

IMAGE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.matplotlib__matplotlib-20859:latest"
TEST = "lib/matplotlib/tests/test_legend.py::test_warn_big_data_best_loc"
N = 20

script = f"""
cd /testbed && source /opt/miniconda3/bin/activate && conda activate testbed
pass=0; fail=0
for i in $(seq 1 {N}); do
  python -m pytest {TEST} -x -q --no-header 2>&1 | tail -1 | grep -q 'passed' && pass=$((pass+1)) || fail=$((fail+1))
done
echo "Result: $pass/{N} passed, $fail/{N} failed"
"""

if __name__ == "__main__":
    print(f"Running {TEST}")
    print(f"  {N} times in unmodified container: {IMAGE}")
    print()

    result = subprocess.run(
        ["docker", "run", "--rm", IMAGE, "bash", "-c", script],
        capture_output=True, text=True,
    )

    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        print(f"\nDocker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)
