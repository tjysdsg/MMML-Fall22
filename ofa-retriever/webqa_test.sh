CUDA_VISIBLE_DEVICES=3 python main.py --test \
--cache_dir ./cache/WebQA_test_data_cache/ \
--dataset_dir ./data/WebQA_test_data/ \
--test_file test.jsonl \
--test_batch_size 45 \
--have_cached_dataset \
--without_image \
--num_workers 4