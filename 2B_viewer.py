"""
Launch the Inspect log viewer for browsing evaluation results.

Opens a local web server at http://127.0.0.1:7575.

Usage:
    python 2B_viewer.py
"""

import subprocess
import sys

LOG_DIR = "./logs"

if __name__ == "__main__":
    print(f"Starting Inspect viewer for {LOG_DIR}")
    print("Open http://127.0.0.1:7575 in your browser")
    print("Press Ctrl+C to stop\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "inspect_ai", "view", "--log-dir", LOG_DIR],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nViewer stopped.")
