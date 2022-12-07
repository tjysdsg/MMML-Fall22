CUDA_VISIBLE_DEVICES=2 python main.py \
--train \
--use_fp16 \
--task webqa-finetune-full-data \
--use_wandb \
--model_name ofa-base \
--model_dir ./ofa-base \
--cache_dir ./cache/WebQA_full_data_cache/ \
--dataset_dir ./data/WebQA_full_data/ \
--gradient_accumulation_step 128 \
--train_file train.jsonl \
--val_file val.jsonl \
--num_epochs 4 \
--evaluation_steps 25 \
--question_max_length 50 \
--fact_max_length 100 \
--answer_max_length 1 \
--dev_batch_size 1 \
--train_batch_size 1 \
--real_batch_size 1 \
--train_choice_num 16 \
--val_choice_num 32 \
--learning_rate 3e-3 \
--have_cached_dataset