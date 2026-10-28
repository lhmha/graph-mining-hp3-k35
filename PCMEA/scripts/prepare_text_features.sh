#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {FB15K_DB15K|FB15K_YAGO15K} PATH_TO_MMKb_REPO" >&2
  exit 2
fi

dataset="$1"
mmkb_root="$2"
python_bin="${PYTHON_BIN:-python3}"
dataset_dir="data/mmkb-datasets/${dataset}"

case "${dataset}" in
  FB15K_DB15K)
    second_graph="DB15K"
    ;;
  FB15K_YAGO15K)
    second_graph="YAGO15K"
    ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    exit 2
    ;;
esac

test -d "${dataset_dir}"
test -f "${mmkb_root}/FB15K/FB15K_EntityTriples.txt"
test -f "${mmkb_root}/${second_graph}/${second_graph}_EntityTriples.txt"

"${python_bin}" data_process/relation_text_expand.py \
  --ent-ids "${dataset_dir}/ent_ids_1" "${dataset_dir}/ent_ids_2" \
  --entity-triples \
    "${mmkb_root}/FB15K/FB15K_EntityTriples.txt" \
    "${mmkb_root}/${second_graph}/${second_graph}_EntityTriples.txt" \
  --output "${dataset_dir}/relation_text.json"

"${python_bin}" data_process/PLM_embedding.py \
  --kind attribute --model roberta-base \
  --attribute-files "${dataset_dir}/training_attrs_1" "${dataset_dir}/training_attrs_2" \
  --ent-ids "${dataset_dir}/ent_ids_1" "${dataset_dir}/ent_ids_2" \
  --output "${dataset_dir}/attribute_roberta_base_feature_all.pkl"

"${python_bin}" data_process/PLM_embedding.py \
  --kind relation --model t5-base \
  --input-json "${dataset_dir}/relation_text.json" \
  --output "${dataset_dir}/relation_t5_feature_all.pkl"

"${python_bin}" tools/verify_data.py "${dataset}"
