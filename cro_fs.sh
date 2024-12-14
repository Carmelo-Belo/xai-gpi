#!/bin/bash

# Cycle through the different number of clusters and different possible models for FS fitness function
n_clusters=(10 11 12)
models=('pi-lgbm')
n_vars=8
n_idxs=9
basins=("GLB" "NA" "NEP" "NWP" "NI" "SI" "SP")

for basin in "${basins[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        for model in "${models[@]}"; do
            # Run the FS 5 times for each configuration
            for i in {1..5}; do
                # Simulation with normal clusterization
                output_folder="test${i}_${model}_nc${n_cluster}_nv${n_vars}_nd${n_idxs}"
                folder="results/${basin}/${output_folder}"
                if [ -d $folder ]; then
                    echo "The folder exists for $basin, $n_cluster normal cluster, $model model, and test $i"
                else
                    echo "Running FS for $basin with $n_cluster normal clusters, $model model, and test $i"
                    python3 CRO_Spatiotemporal_FS_PI.py --basin $basin --anomaly_clustering 'n' --n_clusters $n_cluster --n_vars $n_vars --n_idxs $n_idxs --model_kind $model --output_folder $output_folder
                fi
                # Simulation with anomaly clusterization
                output_folder="test${i}_${model}_Anc${n_cluster}_nv${n_vars}_nd${n_idxs}"
                folder="results/${basin}/${output_folder}"
                if [ -d $folder ]; then
                    echo "The folder exists for $basin, $n_cluster anomaly cluster, $model model, and test $i"
                else
                    echo "Running FS for $basin with $n_cluster anomaly clusters, $model model, $model model, and test $i"
                    python3 CRO_Spatiotemporal_FS_PI.py --basin $basin --anomaly_clustering 'y' --n_clusters $n_cluster --n_vars $n_vars --n_idxs $n_idxs --model_kind $model --output_folder $output_folder
                fi
            done
        done
    done
done