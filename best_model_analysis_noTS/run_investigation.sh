#!/bin/bash

# Cycle through the different percentage of selection
basin="SI"
n_clusters=10

# # Run investigation for different selection percentages
# sel_percs=("50" "60" "70" "75" "80" "90")

# for perc in "${sel_percs[@]}"; do
#     run_name="selfeat${perc}_top20_nc${n_clusters}_nv8_nd9_noTS"
#     echo "Running run investigation for $basin, $perc% selection"
#     python3 run_investigation.py --basin $basin --run_name $run_name
# done

# Run investigation for specific best simulations
# NEP -> 60, 78, 87
# simul_numbers=(60 78 87)
# NA -> 3, 14, 61
# simul_numbers=(3 14 61)
# SI -> 12, 51, 82
simul_numbers=(12 51 82)

for simul_num in "${simul_numbers[@]}"; do
    run_name="test${simul_num}_linreg_nc${n_clusters}_nv8_nd9_noTS"
    echo "Running run investigation for $basin, simulation number $simul_num"
    python3 run_investigation.py --basin $basin --run_name $run_name
done
