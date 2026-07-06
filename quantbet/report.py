"""Analyze the live paper-trading ledger: PnL, hit rate, and — most
importantly — calibration: how often did bets win compared with what the
model believed at bet time?
"""

from collections import defaultdict

from .ledger import load_ledger


def ledger_stats(history: list[dict]) -> dict:
    settled = [b for b in history if b.get("status") in ("won", "lost")]
    if not settled:
        return {"settled": 0}
    staked = sum(float(b["stake"]) for b in settled)
    profit = sum(float(b["profit"]) for b in settled)
    wins = sum(1 for b in settled if b["status"] == "won")
    return {
        "settled": len(settled),
        "pending": sum(1 for b in history if b.get("status") == "pending"),
        "refunded": sum(1 for b in history if b.get("status") == "refunded"),
        "wins": wins,
        "hit_rate": wins / len(settled),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else 0.0,
        "avg_odds": sum(float(b["odds_taken"]) for b in settled) / len(settled),
        "avg_model_prob": sum(float(b["model_prob"]) for b in settled) / len(settled),
    }


def calibration_buckets(history: list[dict], width: float = 0.10) -> list[dict]:
    """Group settled bets by claimed probability and compare with reality."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for b in history:
        if b.get("status") in ("won", "lost"):
            buckets[int(float(b["model_prob"]) / width)].append(b)

    rows = []
    for key in sorted(buckets):
        bets = buckets[key]
        rows.append(
            {
                "range": f"{key * width:.0%}-{(key + 1) * width:.0%}",
                "n": len(bets),
                "claimed": sum(float(b["model_prob"]) for b in bets) / len(bets),
                "actual": sum(1 for b in bets if b["status"] == "won") / len(bets),
            }
        )
    return rows


def print_report() -> None:
    history = load_ledger()
    stats = ledger_stats(history)
    if not stats.get("settled"):
        print("No settled bets in the ledger yet.")
        return

    print("\n--- LIVE PAPER-TRADING REPORT ---")
    print(f"Settled bets:    {stats['settled']} ({stats['pending']} pending, "
          f"{stats['refunded']} refunded)")
    print(f"Hit rate:        {stats['hit_rate']:.1%}")
    print(f"Avg odds taken:  {stats['avg_odds']:.2f}")
    print(f"Avg model prob:  {stats['avg_model_prob']:.1%}")
    print(f"Turnover:        ${stats['staked']:.2f}")
    print(f"Profit/Loss:     ${stats['profit']:+.2f}")
    print(f"ROI:             {stats['roi']:+.1%}")

    print("\nCalibration (claimed win prob vs realized win rate):")
    print(f"{'bucket':>10} {'n':>5} {'claimed':>9} {'actual':>8} {'gap':>8}")
    for row in calibration_buckets(history):
        gap = row["actual"] - row["claimed"]
        print(f"{row['range']:>10} {row['n']:>5} {row['claimed']:>8.1%} "
              f"{row['actual']:>7.1%} {gap:>+7.1%}")


if __name__ == "__main__":
    print_report()
