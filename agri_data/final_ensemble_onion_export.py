"""Host-side Parquet exporter for FINAL_ENSEMBLE_LOSS_ONION_V2.

This intentionally uses pandas/pyarrow only; it does not initialize Torch or
load fit checkpoints.
"""
from __future__ import annotations

import json

from .final_ensemble_onion import export_forecasts


def main() -> int:
    result = export_forecasts()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
