python gen_offline_data.py \
  --data_dir ../data/gt_data-train_10cats_train_data-pushing \
  --data_fn ../stats/train_10cats_train_data_list.txt \
  --primact_types pushing \
  --num_processes [?] \
  --num_epochs 150 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt

python gen_offline_data.py \
  --data_dir ../data/gt_data-train_10cats_test_data-pushing \
  --data_fn ../stats/train_10cats_test_data_list.txt \
  --primact_types pushing \
  --num_processes [?] \
  --num_epochs 10 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt

python gen_offline_data.py \
  --data_dir ../data/gt_data-test_5cats-pushing \
  --data_fn ../stats/test_5cats_data_list.txt \
  --primact_types pushing \
  --num_processes [?] \
  --num_epochs 10 \
  --ins_cnt_fn ../stats/ins_cnt_5cats.txt

