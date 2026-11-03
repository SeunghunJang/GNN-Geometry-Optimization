#!/usr/bin/env bash
# Quick start: bash scripts/run_demo.sh [B3D BDim ... | all]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ "$#" -eq 1 ] && [ "$1" = "all" ]; then
  set -- B3D E3D Z3D BDim EDim ZDim
elif [ "$#" -eq 0 ]; then
  set -- B3D
fi
python -u scripts/run_demo.py --models "$@" --execute
