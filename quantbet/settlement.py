"""Grade pending ledger bets against final match results."""

import logging

from .api_client import _stats_get
from .config import HISTORY_FILE
from .ledger import load_ledger, save_ledger

log = logging.getLogger(__name__)

FINISHED = ("FT", "AET", "PEN")
VOIDED = ("CANC", "PSTP", "ABD")


def grade_bet(bet_placed: str, actual_result: str, stake: float, odds: float) -> tuple[str, float]:
    """Pure grading rule -> (status, profit).

    Handles straight 1X2 picks plus Draw-No-Bet variants ("Home (DNB)"),
    where a draw refunds the stake.
    """
    if actual_result == "Void":
        return "refunded", 0.0
    if "(DNB)" in bet_placed and actual_result == "Draw":
        return "refunded", 0.0
    picked_side = bet_placed.replace(" (DNB)", "")
    if picked_side == actual_result:
        return "won", round(stake * (odds - 1.0), 2)
    return "lost", -stake


def fetch_match_result(home_id: int, away_id: int, bet_date: str, bet_timestamp: str) -> str:
    """Look up the final result of home_id vs away_id.

    Returns "Home" / "Draw" / "Away" / "Void" / "Pending".
    """
    target_date = bet_date[:10] if bet_date else ""
    params = {"team": home_id, "date": target_date} if target_date else {"team": home_id, "last": 15}

    try:
        data = _stats_get("fixtures", params)
    except Exception as e:
        log.error("Stats API error while settling: %s", e)
        return "Pending"

    for fixture in data.get("response", []):
        if fixture["teams"]["away"]["id"] != away_id:
            continue

        # Without an exact fixture date we could match an old head-to-head
        # meeting; only accept fixtures on/after the day the bet was logged.
        if not target_date and bet_timestamp:
            if fixture["fixture"]["date"][:10] < bet_timestamp[:10]:
                continue

        status = fixture["fixture"]["status"]["short"]
        if status in FINISHED:
            h_g = fixture["goals"]["home"]
            a_g = fixture["goals"]["away"]
            if h_g is None or a_g is None:
                return "Pending"
            if h_g > a_g:
                return "Home"
            if a_g > h_g:
                return "Away"
            return "Draw"
        if status in VOIDED:
            return "Void"
        return "Pending"

    return "Pending"


def run_settlement() -> None:
    history = load_ledger()
    if not history:
        log.warning("No ledger found at %s — nothing to settle.", HISTORY_FILE)
        return

    print("\n--- SETTLEMENT ENGINE ---")
    updates = 0
    session_pnl = 0.0

    for bet in history:
        if bet.get("status") != "pending":
            continue
        result = fetch_match_result(
            bet["home_id"], bet["away_id"], bet.get("date", ""), bet.get("timestamp", "")
        )
        if result == "Pending":
            continue

        bet["actual_result"] = result  # kept for the retrainer
        status, profit = grade_bet(bet["bet_placed"], result, float(bet["stake"]), float(bet["odds_taken"]))
        bet["status"] = status
        bet["profit"] = profit
        session_pnl += profit
        updates += 1
        tag = {"won": "WIN", "lost": "LOSS", "refunded": "REFUND"}[status]
        print(f"[{tag}] {bet['home_team']} vs {bet['away_team']} ({profit:+.2f})")

    if updates:
        save_ledger(history)
        print(f"\n[SUCCESS] {updates} bet(s) settled. Session PnL: ${session_pnl:.2f}")
    else:
        print("[INFO] No pending matches have finished yet.")

    settled = [b for b in history if b.get("status") in ("won", "lost")]
    total_profit = sum(float(b["profit"]) for b in settled)
    total_staked = sum(float(b["stake"]) for b in settled)
    in_flight = sum(float(b["stake"]) for b in history if b.get("status") == "pending")

    print("\n--- LIFETIME PAPER-TRADING SUMMARY ---")
    print(f"Settled bets:      {len(settled)}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    if total_staked:
        print(f"ROI on turnover:   {100 * total_profit / total_staked:.2f}%")
    print(f"Money in flight:   ${in_flight:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_settlement()
