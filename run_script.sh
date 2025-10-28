#!/bin/bash
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train OPC &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train SST &
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --cell_type_to_train MGE_dev
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --cell_type_to_train MGE_dev

# PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_decoder.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --vis_pred --metric_only
# PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_decoder.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize --vis_all_embeds --metric_only

# kl coefficients to test out: 0.0, 0.1, 0.2, 0.5, 0.8, 1.0

# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py\
#  -d herring --hvgs -s remove_recovery --normalize --kl_coeff 1.0 &

# PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py\
#  -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.0 --vis_all_embeds --pretrain_only
# echo "Running train_per_cell_type.py with kl_coeff=0.001 and freezing enc, dec weights"
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#     -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 --freeze_enc_dec

flags="\
--freeze_enc_dec \
--grad_norm \
--beta 0.01 \
"

echo "Running train_per_cell.py with kl_coeff=0.001, vel_reg is True, Gradnorm is True"
PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
    -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
    $flags

echo "Running benchmark/benchmark_encoder.py with kl_coeff=0.001, vel_reg is True, Gradnorm is True"
PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
    -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
    $flags \
    --vis_all_embeds --vis_pred --metric_only

# echo "Running train_per_cell.py with kl_coeff=0.001 and freezing enc, dec weights, beta=10"
# PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#     -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#     --freeze_enc_dec --beta 10

# echo "Running benchmark_encoder.py with kl_coeff=0.001 and freezing enc, dec weights, beta=10"
# PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#     -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#     --vis_pred --freeze_enc_dec --beta 10 --metric_only

# for KL in 0.00001
# do
#     echo "Running train_per_cell_type.py with kl_coeff=0.001, full_train_kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#         --full_train_kl_coeff ${KL} --adjusted_full_train
#     echo "Running benchmark_encoder.py with kl_coeff=0.001, full_train_kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#         --vis_all_embeds --full_train_kl_coeff ${KL} --adjusted_full_train
# done


# for lr in 1e-3
# do
#     for ftlr in 1e-4
#     do
#         echo "Running train_per_cell_type.py with frozen enc/dec, kl_coeff=0.001, lr=${lr}, finetune_lr=${ftlr}"
#         PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#             -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#             --lr ${lr} --finetune_lr ${ftlr} --beta 10

#         echo "Running benchmark_encoder.py (vis_all_embeds) with kl_coeff=0.001, lr=${lr}, finetune_lr=${ftlr}"
#         PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#             -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#             --lr ${lr} --finetune_lr ${ftlr} --vis_all_embeds --beta 10

#         echo "Running benchmark_encoder.py (vis_pred) with kl_coeff=0.001, lr=${lr}, finetune_lr=${ftlr}"
#         PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#             -d herring --hvgs -s remove_recovery --normalize --kl_coeff 0.001 \
#             --lr ${lr} --finetune_lr ${ftlr} --vis_pred --beta 10
#     done
# done


# for KL in 0.001
# do
#     echo "Running train_per_cell_type.py with kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff ${KL}

#     echo "Running benchmark_encoder.py with kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff ${KL} \
#         --vis_all_embeds

#     echo "Finished KL: ${KL}"
#     echo "--------------------------------"
# done



# for KL in 0.001 0.005
# do
#     echo "Running train_per_cell_type.py with kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/train_per_cell_type.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff ${KL}

#     echo "Running benchmark_encoder.py with kl_coeff=${KL}"
#     PYTHONPATH=".:$PYTHONPATH" python benchmark/benchmark_encoder.py \
#         -d herring --hvgs -s remove_recovery --normalize --kl_coeff ${KL} \
#         --vis_all_embeds --pretrain_only

#     echo "Finished KL: ${KL}"
#     echo "--------------------------------"
# done

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
