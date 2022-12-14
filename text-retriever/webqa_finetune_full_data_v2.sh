export CUDA_VISIBLE_DEVICES=0
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29513 main.py \
--train \
--use_fp16 \
--task webqa-finetune-full-data \
--use_wandb \
--cache_dir ./cache/WebQA_full_data_cache/ \
--dataset_dir ./data/WebQA_full_data/ \
--gradient_accumulation_step 128 \
--train_file train.jsonl \
--val_file val.jsonl \
--num_epochs 6 \
--evaluation_steps 200 \
--max_length 300 \
--choice_num 16 \
--have_cached_dataset \
