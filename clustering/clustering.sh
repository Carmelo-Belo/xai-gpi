#!/bin/bash

# Cycle through the different basins and number of clusters
# basins=("NA" "NEP" "NWP" "NI" "SI" "SP")
basins=("SP")
n_clusters=(5 6 7 8 9 10 11 12)

for basin in "${basins[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        # Normal Clustering
        python3 clustering.py --n_clusters $n_cluster --basin $basin
        echo "Normal Clustering for $basin with $n_cluster clusters is done"
        # Deseasonalized Clustering
        python3 clustering.py --n_clusters $n_cluster --basin $basin --remove_seasonality 'y'
        echo "Deseasonalized Clustering for $basin with $n_cluster clusters is done"
        # Detrended Clustering
        python3 clustering.py --n_clusters $n_cluster --basin $basin --remove_trend 'y'
        echo "Detrended Clustering for $basin with $n_cluster clusters is done"
        # Detrended and Deseasonalized Clustering
        python3 clustering_noTS.py --n_clusters $n_cluster --basin $basin
        echo "Detrended and deseasonalized Clustering for $basin with $n_cluster clusters is done"
    done
done