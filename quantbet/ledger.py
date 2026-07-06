"""Paper-trading ledger: an append-only JSON file of every recommended bet.

Each record stores a snapshot of the exact model inputs at bet time
(`features`), so the nightly retrainer can learn from the features the model
actually saw instead of re-fetching *current* form for a match played weeks
ago (which is look-ahead leakage — the old implementation did exactly that).
"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import HISTORY_FILE


def load_ledger(path: Path = HISTORY_FILE) -> list[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return json.load(f)


def _atomic_write(path: Path, history: list[dict]) -> None:
    """Write via a temp file + rename so a crash mid-write can't corrupt the
    ledger (it holds the full live track record)."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(history, f, indent=4)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def save_ledger(history: list[dict], path: Path = HISTORY_FILE) -> None:
    _atomic_write(path, history)


def append_bet(
    bet: dict,
    stake: float,
    feature_snapshot: dict | None = None,
    path: Path = HISTORY_FILE,
) -> dict:
    history = load_ledger(path)
    record = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "home_team": str(bet["h_name"]),
        "home_id": int(bet["h_id"]),
        "away_team": str(bet["a_name"]),
        "away_id": int(bet["a_id"]),
        "league_id": int(bet["l_id"]),
        "date": bet.get("date", ""),
        "bet_placed": str(bet["bet"]),
        "odds_taken": float(bet["odds"]),
        "stake": float(stake),
        "model_prob": float(bet["prob"]),
        "model_ev": float(bet["ev"]),
        "status": "pending",
        "profit": 0.0,
    }
    if feature_snapshot is not None:
        record["features"] = feature_snapshot
    history.append(record)
    _atomic_write(path, history)
    return record
