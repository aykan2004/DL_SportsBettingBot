# Data provenance

## soccer_data_full.csv (training set)

~4.5k finished matches (Apr 2023 – Feb 2026) across the five leagues in
`config.LEAGUE_MAP`, collected from [api-football](https://www.api-football.com/)
v3 `/fixtures`. One row per match:

| Column | Meaning |
|---|---|
| `date` | Kickoff, UTC |
| `league_id` | api-football league id (39 EPL, 2 UCL, 3 UEL, 203 Süper Lig, 71 Série A BR) |
| `home_team`, `away_team` | api-football team names (also the embedding vocabulary keys) |
| `home_goals`, `away_goals` | Full-time score |
| `result` | 0 = home win, 1 = draw, 2 = away win |
| `games_played_*`, `points_*` | Season-to-date standings context |
| `home_form`, `away_form` | League-weighted points over the previous 5 matches (weights in `config.LEAGUE_WEIGHTS`) |
| `home_gd`, `away_gd` | Average goal difference over the previous 5 matches |
| `home_importance`, `away_importance` | Stakes heuristic in [0,1] (title/relegation/European race proximity); 0.5 = unknown/neutral |

Rolling columns are computed from matches strictly before each row.

## historical_dataset_2024.json (backtest set)

142 matches (Sep–Oct 2024) that additionally carry a **pre-kickoff (T−2h)
1X2 odds snapshot** from The Odds API's paid `/historical` endpoint (first
listed EU bookmaker). Rebuild with `python -m quantbet build-dataset` —
note it spends historical-odds credits. Labels use the same 0/1/2 encoding.

## bet_history.json (live paper-trading ledger)

Every bet recommended and logged by the live pipeline since 2026-02-19, with
model probability, EV and odds at bet time, settlement status and PnL. Bets
logged by v2 also carry a `features` snapshot (exact model inputs at bet
time) used by the nightly retrainer. This file is the evidence behind the
README's forward-test numbers; `python -m quantbet report` recomputes them.
