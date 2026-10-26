#!/bin/bash
set -euo pipefail

OLD_PREFIX=/old/location/of/datasets
NEW_PREFIX="$(pwd)/datasets"

for csv_path in datasets/*/index*.csv; do
  python scripts/data/rewrite_index_paths.py \
    --in_csv "${csv_path}" \
    --inplace \
    --rule "${OLD_PREFIX}" "${NEW_PREFIX}"
done
