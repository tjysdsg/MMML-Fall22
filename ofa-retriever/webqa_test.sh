export CUDA_VISIBLE_DEVICES=3
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29528 main.py \
--test \
--dataset_dir ./data/WebQA_test_data/ \
--test_file test.jsonl \
--test_batch_size 50