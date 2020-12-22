python gen_offline_data.py \
  --data_dir ../data/gt_data-single_cabinet_train_data-pushing \
  --data_fn ../stats/single_cabinet.txt \
  --primact_types pushing \
  --num_processes 8 \
  --num_epochs 20 \
  --ins_cnt_fn ../stats/ins_cnt_single_cabinet.txt

python gen_offline_data.py \
  --data_dir ../data/gt_data-single_cabinet_test_data-pushing \
  --data_fn ../stats/single_cabinet.txt \
  --primact_types pushing \
  --num_processes 8 \
  --num_epochs 2 \
  --ins_cnt_fn ../stats/ins_cnt_single_cabinet.txt

