#!/bin/bash

# Cycle through the different basins and number of clusters
basins=("GLB" "NEP" "NWP" "NA" "NI" "SP" "SI")
n_clusters=(5 6 7 8 9 10 11 12)
filenames=("averages_mpi.csv" "centroids_mpi.csv" "labels_mpi.csv")

for basin in "${basins[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        for filename in "${filenames[@]}"; do
            # normal clusters
            dkrz_path="/home/simul6/Documentos/Filippo/FS_TCG/data/${basin}_${n_clusters}clusters/"
            sshpass -p 'aloha123' scp /Users/huripari/Documents/PhD/TCs_Genesis/FS_TCG/data/${basin}_${n_clusters}clusters/${filename} "simul6@192.168.77.128:$dkrz_path"
            echo "Uploaded $filename for $basin with $n_cluster clusters"
            # anomaly clusters
            dkrz_path="/home/simul6/Documentos/Filippo/FS_TCG/data/${basin}_${n_clusters}clusters_anomaly/"
            sshpass -p 'aloha123' scp /Users/huripari/Documents/PhD/TCs_Genesis/FS_TCG/data/${basin}_${n_clusters}clusters_anomaly/${filename} "simul6@192.168.77.128:$dkrz_path"
            echo "Uploaded $filename for $basin with $n_cluster anomaly clusters"
            # deseasonalized clusters
            dkrz_path="/home/simul6/Documentos/Filippo/FS_TCG/data/${basin}_${n_clusters}clusters_deseason/"
            sshpass -p 'aloha123' scp /Users/huripari/Documents/PhD/TCs_Genesis/FS_TCG/data/${basin}_${n_clusters}clusters_deseason/${filename} "simul6@192.168.77.128:$dkrz_path"
            echo "Uploaded $filename for $basin with $n_cluster deseasonalized clusters"
        done
    done
done