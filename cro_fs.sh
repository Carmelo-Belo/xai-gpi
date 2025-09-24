#!/bin/bash

# Cycle through the different number of clusters and different possible models for FS fitness function
n_clusters=(5 6 7 8 9 10 11 12)
models=('linreg' 'lgbm' 'pi-lgbm')
n_vars=8
n_idxs=9
basins=("NA" "NEP" "NWP" "NI" "SI" "SP")

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
                    # run different scripts depending on the model used for the fitnesse function
                    if [ "$model" == "pi-lgbm" ]; then
                        python3 CRO_Spatiotemporal_FS_PI.py --basin $basin --n_clusters $n_cluster --remove_trend 'n' --remove_seasonality 'n' --n_vars $n_vars --n_idxs $n_idxs --model_kind $model --output_folder $output_folder
                    else
                        python3 CRO_Spatiotemporal_FS.py --basin $basin --n_clusters $n_cluster --remove_trend 'n' --remove_seasonality 'n' --n_vars $n_vars --n_idxs $n_idxs --model_kind $model --output_folder $output_folder
                    fi
                fi
            done
        done
    done
done