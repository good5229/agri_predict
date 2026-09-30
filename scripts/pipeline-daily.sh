#!/bin/bash
set -u

PROJECT=/Users/bellhundred/agri_predict
cd "$PROJECT"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--dry-run" ]; }; then
  echo "usage: pipeline-daily.sh [--dry-run]" >&2
  exit 64
fi

set +e
if [ "$#" -eq 1 ]; then
  /bin/bash "$PROJECT/scripts/collect-daily.sh" --dry-run
else
  /bin/bash "$PROJECT/scripts/collect-daily.sh"
fi
COLLECTION_EXIT=$?
if [ "$#" -eq 1 ]; then
  /bin/bash "$PROJECT/scripts/train-daily.sh" --collection-exit-code "$COLLECTION_EXIT" --dry-run
else
  /bin/bash "$PROJECT/scripts/train-daily.sh" --collection-exit-code "$COLLECTION_EXIT"
fi
TRAINING_EXIT=$?
set -e

if [ "$TRAINING_EXIT" -ne 0 ]; then
  exit "$TRAINING_EXIT"
fi
exit 0
