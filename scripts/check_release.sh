#!/usr/bin/env bash
# Pre-upload static + data-fixture checks. Does not train or modify model files.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo '=== Source SHA256 (10 originals) ==='
sha256sum -c SOURCE_SHA256.txt
echo '=== Unit/integration tests ==='
python -m unittest discover -s tests -v
echo '=== Small demo data check ==='
python scripts/run_demo.py --inspect
echo '=== All six training commands (dry-run) ==='
python scripts/run_demo.py --models B3D E3D Z3D BDim EDim ZDim
printf '\nPASS: static pre-upload checks complete; GPU training test is separate.\n'
