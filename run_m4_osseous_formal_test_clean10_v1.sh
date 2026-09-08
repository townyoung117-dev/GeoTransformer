#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"
exec "${PYTHON:-python}" -B experiments/geotransformer.pointct.baseline_v1/run_m4_osseous_formal_test.py "$@"
