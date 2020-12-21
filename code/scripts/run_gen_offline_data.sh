python gen_offline_data.py \
  --data_dir ../data/gt_data-single_cabinet_pushing \
  --data_fn ../stats/single_cabinet.txt \
  --primact_types pushing \
  --num_processes 8 \
  --num_epochs 1000 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt

