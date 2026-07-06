"""Single command-line entry point.

Usage:
    python -m quantbet slip --profile safe      # generate + optionally log bets
    python -m quantbet slip --profile value --no-input
    python -m quantbet settle                   # grade pending ledger bets
    python -m quantbet retrain                  # nightly fine-tune
    python -m quantbet train                    # full retrain from the CSV
    python -m quantbet backtest                 # historical odds backtest
    python -m quantbet report                   # live ledger PnL + calibration
"""

import argparse
import logging


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="quantbet", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    slip_p = sub.add_parser("slip", help="scan fixtures and build a bet slip")
    slip_p.add_argument("--profile", choices=["safe", "value"], default="safe")
    slip_p.add_argument("--no-input", action="store_true", help="print the slip without prompting")

    sub.add_parser("settle", help="grade pending bets in the ledger")
    sub.add_parser("retrain", help="fine-tune the model on newly settled bets")
    sub.add_parser("train", help="train the model from scratch on the historical CSV")
    sub.add_parser("backtest", help="run the historical odds backtest")
    sub.add_parser("baseline", help="fit the classical Poisson baseline on the same holdout")
    sub.add_parser("report", help="live ledger PnL and calibration report")
    sub.add_parser("build-dataset", help="(re)build the historical odds dataset — costs API credits")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "slip":
        from .slip import run
        from .strategy import PROFILES

        run(PROFILES[args.profile], interactive=not args.no_input)
    elif args.command == "settle":
        from .settlement import run_settlement

        run_settlement()
    elif args.command == "retrain":
        from .retrain import retrain

        retrain()
    elif args.command == "train":
        from .train import train

        train()
    elif args.command == "backtest":
        from .backtest import run_backtest

        run_backtest()
    elif args.command == "baseline":
        from .baselines import run_baseline

        run_baseline()
    elif args.command == "report":
        from .report import print_report

        print_report()
    elif args.command == "build-dataset":
        from .build_dataset import build

        build()


if __name__ == "__main__":
    main()
