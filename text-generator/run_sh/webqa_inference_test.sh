CUDA_VISIBLE_DEVICES=3 python ../main.py --inference \
--test_file ../data/WebQA_test_data/retrieved_test.jsonl \
--retrieved_result_file ../data/WebQA_test_data/submission_test.json \
--result_file ../data/WebQA_test_data/final_submission_test.json \