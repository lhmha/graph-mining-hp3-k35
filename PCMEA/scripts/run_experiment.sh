#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {FB15K_DB15K|FB15K_YAGO15K} {0.2|0.5|0.8} [seed]" >&2
  exit 2
fi

dataset="$1"
rate="$2"
seed="${3:-42}"
python_bin="${PYTHON_BIN:-python3}"
gpu="${GPU_ID:-0}"

case "${dataset}" in
  FB15K_DB15K|FB15K_YAGO15K) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac
case "${rate}" in
  0.2|0.5|0.8) ;;
  *) echo "Unsupported alignment rate: ${rate}" >&2; exit 2 ;;
esac

rate_tag="${rate/./}"
run_name="full_${dataset,,}_${rate_tag}_seed${seed}"
checkpoint_dir="checkpoints/${run_name}"
save_dir="artifacts/${run_name}"
log_file="logs/${run_name}.log"
mkdir -p "${checkpoint_dir}" "${save_dir}" logs

"${python_bin}" tools/verify_data.py "${dataset}"

resume_args=()
tee_args=("${log_file}")
if [[ -f "${checkpoint_dir}/latest.pt" ]]; then
  resume_args+=(--resume)
  tee_args=(-a "${log_file}")
  echo "Resuming from ${checkpoint_dir}/latest.pt"
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

set -o pipefail
/usr/bin/time -v "${python_bin}" -u src/run_PCMEA.py \
  --file_dir "data/mmkb-datasets/${dataset}" \
  --rate "${rate}" --lr .0005 --epochs 4500 \
  --hidden_units 300,300,300 --check_point 50 --bsize 512 \
  --il --il_start 500 --semi_learn_step 5 --stage2_start 500 \
  --momentum_update_span 20 --csls --csls_k 3 --seed "${seed}" \
  --tau 0.1 --tau2 4.0 --structure_encoder gat \
  --img_dim 100 --attr_dim 100 --inner_view_num 6 \
  --save_path "${save_dir}" --checkpoint_dir "${checkpoint_dir}" \
  "${resume_args[@]}" 2>&1 | tee "${tee_args[@]}"
