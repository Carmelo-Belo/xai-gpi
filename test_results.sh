#!/bin/bash

# Cycle through the different basins and temporal horizon
n_clusters=(5 6 7 8 9 10 11 12)
basins=("GLB" "NA" "NEP" "NWP" "NI")
model="pi-lgbm"
n_var=8
n_idx=9

# Cycle through the different basins
for n_cluster in "${n_clusters[@]}"; do
  for basin in "${basins[@]}"; do
    for i in {1..5}; do
      echo "Number of clusters: $n_cluster, basin $basin"
      python3 test_results_analysis.py --basin $basin --n_clusters $n_cluster --n_vars $n_var --n_idxs $n_idx --model_kind $model --results_folder "test${i}_${model}_nc${n_cluster}_nv${n_var}_nd${n_idx}"
      python3 test_results_analysis.py --basin $basin --n_clusters $n_cluster --n_vars $n_var --n_idxs $n_idx --model_kind $model --results_folder "test${i}_${model}_Anc${n_cluster}_nv${n_var}_nd${n_idx}"
    done
  done
done