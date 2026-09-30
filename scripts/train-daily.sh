#!/bin/bash
set -euo pipefail

PROJECT=/Users/bellhundred/agri_predict
PYTHON="$PROJECT/.venv/bin/python"
cd "$PROJECT"
exec "$PYTHON" -m agri_data.daily_training "$@"
