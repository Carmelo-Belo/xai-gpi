#!/bin/bash

# Cycle through the different basins and number of clusters
basins=("GLB" "NEP" "NWP" "NA" "NI" "SP" "SI")
n_clusters=(5 6 7 8 9 10 11 12)

for basin in "${basins[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        echo "Building dataset for $basin with $n_cluster normal clusters, and deseasonalized data"
        python3 build_dataset.py --n_clusters $n_cluster --basin $basin --anomaly_clustering 'n' --remove_seasonality 'y'
    done
done