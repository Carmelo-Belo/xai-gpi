#!/bin/bash

# Dictionary for the basins parameters
# ----- NorthEast Pacific -----
# basin="NEP"
# n_clusters=9
# simul_numbers=(60 78 87)
# ----- Northwest Pacific -----
# basin="NWP"
# n_clusters=8
# simul_numbers=(4 25 83)
# ----- North Atlantic --------
# basin="NA"
# n_clusters=12
# simul_numbers=(3 14 61)
# ----- North Indian ----------
# basin="NI"
# n_clusters=9
# simul_numbers=(26 32 45)
# ----- South Indian ----------
# basin="SI"
# n_clusters=10
# simul_numbers=(12 51 82)
# ----- South Pacific ---------
# basin="SP"
# n_clusters=11
# simul_numbers=(8 23 98)

# Cycle through the different percentage of selection
basin="NI"
n_clusters=9
simul_numbers=(26 32 45)

# Run investigation for different selection percentages
sel_percs=("50" "60" "70" "75" "80" "90")
for perc in "${sel_percs[@]}"; do
    run_name="selfeat${perc}_top20_nc${n_clusters}_nv8_nd9_noTS"
    echo "Running run investigation for $basin, $perc% selection"
    python3 run_investigation.py --basin $basin --run_name $run_name
done

# Run investigation for specific best simulations
for simul_num in "${simul_numbers[@]}"; do
    run_name="test${simul_num}_linreg_nc${n_clusters}_nv8_nd9_noTS"
    echo "Running run investigation for $basin, simulation number $simul_num"
    python3 run_investigation.py --basin $basin --run_name $run_name
done

# # Cycle through the different percentage of selection
# basin="SP"
# n_clusters=11
# simul_numbers=(8 23 98)

# # Run investigation for different selection percentages
# sel_percs=("50" "60" "70" "75" "80" "90")
# for perc in "${sel_percs[@]}"; do
#     run_name="selfeat${perc}_top20_nc${n_clusters}_nv8_nd9_noTS"
#     echo "Running run investigation for $basin, $perc% selection"
#     python3 run_investigation.py --basin $basin --run_name $run_name
# done

# # Run investigation for specific best simulations
# for simul_num in "${simul_numbers[@]}"; do
#     run_name="test${simul_num}_linreg_nc${n_clusters}_nv8_nd9_noTS"
#     echo "Running run investigation for $basin, simulation number $simul_num"
#     python3 run_investigation.py --basin $basin --run_name $run_name
# done
