"""Issue immutable onion 3–10 day forecasts using the reviewed D+1/D+2 path."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from . import onion_prospective as op

ROOT = op.ROOT
HORIZONS = tuple(range(3, 11))


def default_output(issue_date: dt.date) -> Path:
    """Keep this independent from the existing first-2026-10-07.json series."""
    return ROOT / "outputs/prospective-onion" / issue_date.strftime("%Y-%m") / "horizons-3-10" / f"{issue_date.isoformat()}.json"


def run(prices_path: Path, db_path: Path, output_path: Path, *, issue_date=None,
        now=None, state_path=None, first_month_issue=False):
    """Delegate all extraction, validation, fitting and no-clobber behavior."""
    return op.run(prices_path, db_path, output_path, issue_date=issue_date, now=now,
                  state_path=state_path, first_month_issue=first_month_issue,
                  horizons=HORIZONS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data/agri.sqlite")
    parser.add_argument("--output", type=Path, help="defaults to a date-keyed 3–10 day artifact")
    parser.add_argument("--issue-date", type=dt.date.fromisoformat)
    parser.add_argument("--state", type=Path, help="first 3–10 day artifact for this month")
    parser.add_argument("--first-issue", action="store_true", help="establish this month's 3–10 day model state")
    args = parser.parse_args(argv)
    issue_date = args.issue_date or dt.datetime.now(op.KST).date()
    if bool(args.state) == bool(args.first_issue):
        parser.error("provide exactly one of --state or --first-issue")
    output = args.output or default_output(issue_date)
    result = run(args.prices, args.db, output, issue_date=issue_date,
                 state_path=args.state, first_month_issue=args.first_issue)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
