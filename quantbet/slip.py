"""Generate a bet slip: scan upcoming fixtures, price them with the model,
compare against market odds and keep whatever clears the strategy filters.

Replaces the old main_safe.py / main_value.py pair, which were 95% identical
copy-paste and had already drifted apart.
"""

import logging
import time

from .api_client import (
    current_season,
    get_match_odds,
    get_team_recent_stats,
    get_upcoming_fixtures,
)
from .config import LEAGUE_MAP, MAPPINGS_FILE, MODEL_FILE, require_api_keys
from .features import NEUTRAL_IMPORTANCE, OUTCOME_NAMES, combine_sequences, make_continuous
from .ledger import append_bet
from .model import Mappings, load_model, predict_proba
from .strategy import StrategyProfile, dedupe_by_match, expected_value, kelly_fraction

log = logging.getLogger(__name__)

API_COURTESY_DELAY = 0.2  # seconds between fixture lookups
MAX_SLIP_SIZE = 10


def _price_fixture(net, mappings: Mappings, fixture: dict, league_id: int) -> list[dict]:
    """Return candidate bets (one per outcome with live odds) for a fixture."""
    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    if home not in mappings.teams or away not in mappings.teams:
        return []

    home_stats = get_team_recent_stats(fixture["teams"]["home"]["id"])
    away_stats = get_team_recent_stats(fixture["teams"]["away"]["id"])
    if home_stats is None or away_stats is None:
        log.info("Skipping %s vs %s: no recent form available.", home, away)
        return []

    (h_form, h_gd), h_hist = home_stats
    (a_form, a_gd), a_hist = away_stats

    # Live match-importance data isn't available from this API tier, so use
    # the neutral prior the model saw for such rows in training.
    cont = make_continuous(
        h_form, a_form, h_gd, a_gd,
        mappings.scaler_mean, mappings.scaler_scale,
        NEUTRAL_IMPORTANCE, NEUTRAL_IMPORTANCE,
    )
    seq = combine_sequences(h_hist, a_hist)

    probs = predict_proba(
        net,
        mappings,
        mappings.teams[home],
        mappings.teams[away],
        mappings.leagues.get(league_id, 0),
        seq,
        cont,
    )

    odds = get_match_odds(LEAGUE_MAP[league_id]["key"], home, away)
    if not odds:
        return []

    snapshot = {"cont": cont.tolist(), "seq": seq}
    candidates = []
    for i, label in enumerate(OUTCOME_NAMES):
        if odds[i] <= 1.01:  # suspended / dead market
            continue
        candidates.append(
            {
                "h_name": home, "a_name": away,
                "h_id": fixture["teams"]["home"]["id"],
                "a_id": fixture["teams"]["away"]["id"],
                "l_id": league_id,
                "league": LEAGUE_MAP[league_id]["name"],
                "date": fixture["fixture"]["date"][:16],
                "match": f"{home} vs {away}",
                "bet": label,
                "prob": float(probs[i]),
                "odds": float(odds[i]),
                "ev": expected_value(float(probs[i]), float(odds[i])),
                "kelly": kelly_fraction(float(probs[i]), float(odds[i])),
                "features": snapshot,
            }
        )
    return candidates


def build_slip(profile: StrategyProfile) -> list[dict]:
    require_api_keys()
    mappings = Mappings.load(MAPPINGS_FILE)
    net = load_model(MODEL_FILE, mappings)

    candidates = []
    for league_id, info in LEAGUE_MAP.items():
        print(f"[INFO] Scanning {info['name']}...")
        season = current_season(league_id)
        for fixture in get_upcoming_fixtures(league_id, season):
            for cand in _price_fixture(net, mappings, fixture, league_id):
                if profile.accepts(cand["prob"], cand["odds"]):
                    candidates.append(cand)
            time.sleep(API_COURTESY_DELAY)

    candidates = dedupe_by_match(candidates)
    candidates.sort(key=lambda c: c[profile.sort_key], reverse=True)
    return candidates[:MAX_SLIP_SIZE]


def print_slip(slip: list[dict], profile: StrategyProfile) -> None:
    print("-" * 60)
    if not slip:
        print(f"[INFO] No bets clear the '{profile.name}' filters right now.")
        return
    for i, bet in enumerate(slip, 1):
        stake_pct = profile.stake_fraction(bet["prob"], bet["odds"]) * 100
        print(f"[{i}] {bet['league']} | {bet['match']} ({bet['date']})")
        print(
            f"    {bet['bet']} @ {bet['odds']:.2f} | model {bet['prob']:.1%} "
            f"| EV {bet['ev']:+.1%} | stake {stake_pct:.2f}% of bankroll"
        )


def log_bets_interactively(slip: list[dict], profile: StrategyProfile) -> None:
    """Prompt the user to paper-log slip entries to the ledger."""
    while True:
        raw = input("\n[INPUT] Enter bet # to log (or 'q' to quit): ").strip().lower()
        if raw == "q":
            return
        if not raw.isdigit() or not (1 <= int(raw) <= len(slip)):
            print(f"[WARN] Enter a number between 1 and {len(slip)}.")
            continue

        bet = slip[int(raw) - 1]
        suggested = round(100 * profile.stake_fraction(bet["prob"], bet["odds"]), 2)
        raw_stake = input(
            f"[INPUT] Stake in $ (suggested: ${suggested:.2f} per $100 bankroll) "
            f"[Enter = {suggested:.2f}]: "
        ).strip()
        try:
            stake = float(raw_stake) if raw_stake else suggested
        except ValueError:
            print(f"[WARN] Not a number; using ${suggested:.2f}.")
            stake = suggested

        append_bet(bet, stake, feature_snapshot=bet["features"])
        print(f"[SUCCESS] Logged {bet['match']} — {bet['bet']} @ {bet['odds']:.2f}, ${stake:.2f}")

        if input("[INPUT] Log another? (y/n): ").strip().lower() != "y":
            return


def run(profile: StrategyProfile, interactive: bool = True) -> list[dict]:
    print(f"\n--- BET SLIP GENERATOR (profile: {profile.name}, 3-day outlook) ---")
    slip = build_slip(profile)
    print_slip(slip, profile)
    if interactive and slip:
        log_bets_interactively(slip, profile)
    return slip
