CUDA_VISIBLE_DEVICES=2 python main.py --test \
--cache_dir ./cache/WebQA_test_data_cache/ \
--image_dir ../../utils/webqa_data/images/ \
--task webqa-finetune-mm \
--dataset_dir ./data/WebQA_test_data/ \
--model_name ofa-base \
--model_dir ./ofa-base \
--test_file test.jsonl \
--test_batch_size 128 \
--num_workers 4 \
--have_cached_dataset