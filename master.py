"""Nightly automation entry point (invoked by cron).

Runs settlement, then retraining, as separate subprocesses so a crash in one
phase cannot corrupt the other. Kept at the repo root under its original name
for crontab compatibility.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python3"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

# Give the network a moment when the machine has just woken up.
time.sleep(15)

print("STARTING DAILY AUTOMATION SEQUENCE...", flush=True)

phases = [
    ("Settlement Engine", ["-m", "quantbet", "settle"]),
    ("ML Retrainer", ["-m", "quantbet", "retrain"]),
]

exit_code = 0
for name, cmd in phases:
    print(f"\n>>> EXECUTING: {name}", flush=True)
    result = subprocess.run([str(PYTHON), *cmd], cwd=ROOT)
    if result.returncode != 0:
        print(f"!!! {name} failed with exit code {result.returncode}", flush=True)
        exit_code = result.returncode
    time.sleep(5)

print("\nDAILY AUTOMATION COMPLETE.", flush=True)
sys.exit(exit_code)
