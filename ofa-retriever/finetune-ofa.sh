export CUDA_VISIBLE_DEVICES=2
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29518 main.py \
--train \
--use_fp16 \
--task webqa-finetune-full-data \
--use_wandb \
--cache_dir ./cache/WebQA_full_data_cache/ \
--dataset_dir ./data/WebQA_full_data/ \
--gradient_accumulation_step 32 \
--train_file train.jsonl \
--val_file val.jsonl \
--num_epochs 2 \
--evaluation_steps 2 \
--max_length 300 \
--dev_batch_size 2 \
--train_batch_size 1 \
--choice_num 16
