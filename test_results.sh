#!/bin/bash

# Cycle through the different basins and temporal horizon
basins=("GLB" "NA" "NEP" "NI" "NWP" "SI" "SP")
n_clusters=(5 6 7 8 9 10 11 12)
models=("pi-lgbm")
n_vars=8
n_idxs=9

# Cycle through the different basins
for basin in "${basins[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        for model in "${models[@]}"; do
            # Run the FS 5 times for each configuration
            for i in {1..5}; do
                echo "Running results analysis for $basin, $n_cluster clusters, $model model, test number $i" 
                output_folder="test${i}_${model}_nc${n_cluster}_nv${n_vars}_nd${n_idxs}"
                python3 test_results_analysis.py --basin $basin --n_clusters $n_cluster --n_vars $n_vars --n_idxs $n_idxs --results_folder $output_folder --model_kind $model
                echo "Running results analysis for $basin, $n_cluster anomaly clusters, $model model, test number $i" 
                output_folder="test${i}_${model}_Anc${n_cluster}_nv${n_vars}_nd${n_idxs}"
                python3 test_results_analysis.py --basin $basin --n_clusters $n_cluster --n_vars $n_vars --n_idxs $n_idxs --results_folder $output_folder --model_kind $model
            done
        done
    done
done