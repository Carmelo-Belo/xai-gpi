#!/bin/bash

# Cycle through the different basins and temporal horizon
n_clusters=(6 8 10 12)
n_vars=(14 28)

# Cycle through the different basins
for n_cluster in "${n_clusters[@]}"; do
  for n_var in "${n_vars[@]}"; do
    echo "Number of clusters: $n_cluster, Number of variables: $n_var"
    python3 test_results_analysis.py --n_clusters $n_cluster --n_vars $n_var --n_idxs 10 --results_folder "test_nc${n_cluster}_nv${n_var}_nd10"
  done
done