"""Bet selection and staking logic: expected value, Kelly sizing, filters.

Pure functions only — no I/O — so every rule that decides whether money is
(paper-)wagered is unit-tested.
"""

from dataclasses import dataclass


def expected_value(prob: float, decimal_odds: float) -> float:
    """EV per unit staked: p * odds - 1."""
    return prob * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Kelly stake as a fraction of bankroll for a single binary bet:
    f* = (b*p - q) / b with b = odds - 1.

    Note: treating each 1X2 outcome as an independent binary bet is a
    simplification (the three outcomes are mutually exclusive); it is
    conservative here because we only ever bet one outcome per fixture.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * prob - (1.0 - prob)) / b)


@dataclass(frozen=True)
class StrategyProfile:
    """A named set of bet filters and staking rules."""

    name: str
    min_prob: float  # minimum model probability for the outcome
    min_ev: float  # minimum expected value per unit staked
    min_odds: float
    max_odds: float
    kelly_multiplier: float  # fractional Kelly (0.5 = half Kelly)
    max_stake_pct: float  # hard cap as a fraction of bankroll
    sort_key: str  # rank candidates by "prob" or "ev"

    def accepts(self, prob: float, decimal_odds: float) -> bool:
        if not (self.min_odds <= decimal_odds <= self.max_odds):
            return False
        return prob >= self.min_prob and expected_value(prob, decimal_odds) >= self.min_ev

    def stake_fraction(self, prob: float, decimal_odds: float) -> float:
        """Fractional-Kelly stake, capped at max_stake_pct of bankroll."""
        return min(
            kelly_fraction(prob, decimal_odds) * self.kelly_multiplier,
            self.max_stake_pct,
        )


# Favours short-priced, high-probability picks with a small required edge.
SAFE = StrategyProfile(
    name="safe",
    min_prob=0.50,
    min_ev=0.05,
    min_odds=1.20,
    max_odds=3.50,
    kelly_multiplier=0.5,
    max_stake_pct=0.05,
    sort_key="prob",
)

# Hunts larger model/market disagreements; sized far more cautiously because
# a big apparent edge is more often model error than market error.
VALUE = StrategyProfile(
    name="value",
    min_prob=0.25,
    min_ev=0.15,
    min_odds=1.20,
    max_odds=8.00,
    kelly_multiplier=0.25,
    max_stake_pct=0.025,
    sort_key="ev",
)

PROFILES = {p.name: p for p in (SAFE, VALUE)}


def dedupe_by_match(candidates: list[dict]) -> list[dict]:
    """Keep only the highest-EV line per fixture. Multiple lines on one match
    are perfectly correlated, so stacking them concentrates risk without
    adding edge."""
    best: dict[str, dict] = {}
    for bet in candidates:
        key = bet["match"]
        if key not in best or bet["ev"] > best[key]["ev"]:
            best[key] = bet
    return list(best.values())
