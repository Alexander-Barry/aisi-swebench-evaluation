"""Patch inspect_evals bash_session timeout from 180 to 210."""
import inspect_evals.swe_bench as m
import os

path = os.path.join(os.path.dirname(m.__file__), "swe_bench.py")
with open(path) as f:
    content = f.read()

if "bash_session(timeout=180)" in content:
    content = content.replace("bash_session(timeout=180)", "bash_session(timeout=210)")
    with open(path, "w") as f:
        f.write(content)
    print("Patched bash_session timeout 180 -> 210")
else:
    print("Already patched or not needed")
