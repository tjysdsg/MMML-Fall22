export CUDA_VISIBLE_DEVICES=1
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 main.py \
--train \
--task webqa-finetune-full-data \
--use_wandb \
--cache_dir ./cache/WebQA_full_data_cache/ \
--dataset_dir ./data/WebQA_full_data/ \
--train_file train.jsonl \
--val_file val.jsonl \
--num_epochs 2 \
--evaluation_steps 500 \
--have_cached_dataset \
