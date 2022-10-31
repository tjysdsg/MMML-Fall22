export CUDA_VISIBLE_DEVICES=1
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 main.py \
--train \
--use_wandb \
--cache_dir ./cache \
--have_cached_dataset \
--dataset_dir ./data/WebQA_sub_data/ \
--train_file train.jsonl \
--val_file val.jsonl \
--evaluation_steps 5 \
