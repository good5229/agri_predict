#!/bin/bash
set -euo pipefail

PROJECT=/Users/bellhundred/agri_predict
PYTHON="$PROJECT/.venv/bin/python"
ENV_FILE=/Users/bellhundred/git-repo/KODATA_Contest/.env

cd "$PROJECT"
exec "$PYTHON" -m agri_data.daily_collection --env-file "$ENV_FILE" "$@"
