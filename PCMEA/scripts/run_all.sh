#!/usr/bin/env bash
set -euo pipefail

seed="${1:-42}"
for dataset in FB15K_DB15K FB15K_YAGO15K; do
  for rate in 0.2 0.5 0.8; do
    bash scripts/run_experiment.sh "${dataset}" "${rate}" "${seed}"
  done
done
