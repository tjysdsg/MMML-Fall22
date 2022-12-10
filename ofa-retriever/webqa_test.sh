CUDA_VISIBLE_DEVICES=0 python main.py --test \
--cache_dir ./cache/WebQA_test_data_cache/ \
--dataset_dir ./data/WebQA_test_data/ \
--model_name ofa-base \
--model_dir ./ofa-base \
--test_file test.jsonl \
--test_batch_size 300 \
--num_workers 4 \
--without_image \
--have_cached_dataset