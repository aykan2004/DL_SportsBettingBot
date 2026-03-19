import subprocess
import time
import sys

PYTHON_PATH = "PASTE YOUR EXACT PYTHON PATH HERE"

print(" STARTING DAILY AUTOMATION SEQUENCE...", flush=True)
print("========================================", flush=True)

# 1. Settle the bets
print("\n>>> EXECUTING Phase 1: Settlement Engine", flush=True)
subprocess.run([PYTHON_PATH, "settle_bets.py"])

# Pause for 5 seconds
time.sleep(5)

# 2. Retrain the model
print("\n>>> EXECUTING Phase 2: ML Retrainer", flush=True)
subprocess.run([PYTHON_PATH, "retrain_bot.py"])

print("\n========================================", flush=True)
print("DAILY AUTOMATION COMPLETE. Going back to sleep.", flush=True)
