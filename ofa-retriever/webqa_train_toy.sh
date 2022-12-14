CUDA_VISIBLE_DEVICES=3 python main.py \
--train \
--use_fp16 \
--task webqa-finetune-full-data \
--use_wandb \
--model_name ofa-base \
--model_dir ./ofa-base \
--cache_dir ./cache/WebQA_toy_data_cache/ \
--dataset_dir ./data/WebQA_toy_data/ \
--gradient_accumulation_step 1 \
--train_file train_toy.jsonl \
--val_file val_toy.jsonl \
--num_epochs 100 \
--evaluation_steps 50 \
--question_max_length 100 \
--fact_max_length 150 \
--answer_max_length 1 \
--dev_batch_size 32 \
--train_batch_size 1 \
--real_batch_size 4 \
--learning_rate 1e-4 \
--warmup_steps 0 \
--without_image \
