#! /bin/bash
#SBATCH --job-name=Apush
#SBATCH --output=scripts/final_run_train_all_v1-10cats-pushing.out
#SBATCH --error=scripts/final_run_train_all_v1-10cats-pushing.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=daerduomkch@gmail.com
#SBATCH --mail-type=ALL

python train_all_v1.py \
    --exp_suffix train_all_v1 \
    --model_version model_all_final \
    --primact_types pushing \
    --data_dir_prefix /checkpoint/kaichun/gt_data_v4 \
    --offline_data_dir /checkpoint/kaichun/gt_data_v4-train_10cats_train_data_pushing \
    --val_data_dir /checkpoint/kaichun/gt_data_v4-train_10cats_test_data_pushing \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../../stats/train_10cats_train_data_list.txt \
    --buffer_max_num 10000 \
    --num_processes_for_datagen 20 \
    --num_interaction_data_offline 50 \
    --num_interaction_data 1 \
    --sample_succ \
    --pretrained_critic_ckpt ./logs/finalexp-model_critic_v1_bce-pushing-None-train_critic_v1_bce_Roff/ckpts/90-network.pth \
    --overwrite

