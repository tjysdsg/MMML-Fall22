CUDA_VISIBLE_DEVICES=0 python main.py \
--train \
--use_fp16 \
--task webqa-finetune-full-data \
--use_wandb \
--cache_dir ./cache/WebQA_full_data_cache/ \
--dataset_dir ./data/WebQA_full_data/ \
--gradient_accumulation_step 128 \
--train_file train.jsonl \
--val_file val.jsonl \
--num_epochs 4 \
--evaluation_steps 200 \
--max_length 300 \
--dev_batch_size 2 \
--train_batch_size 1 \
--choice_num 16 \
--learning_rate 1e-4 \
--have_cached_dataset
