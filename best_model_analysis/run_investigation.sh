#!/bin/bash

# Cycle through the different percentage of selection
basin="SP"
clusters_string="nc7"
# sel_percs=("50" "60" "75" "90")
sel_percs=("60" "75" "90")

for perc in "${sel_percs[@]}"; do
    run_name="selfeat${perc}_top20_${clusters_string}_nv8_nd9"
    echo "Running run investigation for $basin, $perc% selection"
    python3 run_investigation.py --basin $basin --run_name $run_name
done