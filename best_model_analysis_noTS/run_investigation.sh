#!/bin/bash

# Cycle through the different percentage of selection
basin="SI"
n_clusters=10
sel_percs=("50" "60" "70" "75" "80" "90")
# sel_percs=("70" "80")

for perc in "${sel_percs[@]}"; do
    run_name="selfeat${perc}_top20_nc${n_clusters}_nv8_nd9_noTS"
    echo "Running run investigation for $basin, $perc% selection"
    python3 run_investigation.py --basin $basin --run_name $run_name
done