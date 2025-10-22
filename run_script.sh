#!/bin/bash
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring --hvgs --per_cell_type -s remove_recovery &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d wot -s remove_recovery &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_continuous_tps.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_continuous_tps.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation --normalize &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train OPC &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train SST &
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --cell_type_to_train MGE_dev
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train MGE_dev

PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_decoder.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --vis_pred --metric_only


############################## Training for a bunch of different cell types #########################
# Define the two flags
# flags=("--cell_type_to_train" "--cell_type_to_vis")

# # Define the cell types
# cell_types=("Astro" "L2-3_CUX2" "OPC")

# # Make sure logs exist
# mkdir -p logs

# # Ensure Python path is correct
# export PYTHONPATH=".:$PYTHONPATH"

# # Loop over both flags, then over all cell types
# for flag in "${flags[@]}"; do
#     for ct in "${cell_types[@]}"; do
#         echo ">>> Running $flag for $ct ..."
#         # Extract short flag name for logging
#         short_flag=$(echo "$flag" | sed 's/--cell_type_to_//')
#         log_file="logs/${short_flag}_${ct}.out"

#         python benchmark/train_per_cell_type.py -v \
#             -d herring \
#             --hvgs \
#             --per_cell_type \
#             -s remove_recovery \
#             --normalize \
#             "$flag" "$ct" \
#             > "$log_file" 2>&1
#     done
# done

# echo "✅ All runs completed."
