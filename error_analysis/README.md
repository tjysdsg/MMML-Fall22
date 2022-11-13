1. General Error Analysis

    ```bash
    python analyze_retrieval_predictions.py \
      --preds=vlp/result/baseline_predictions_val.json \
      --labels=data\WebQA_train_val.json --split=val
    ```

2. Create multihop subset

    ```bash
    python create_multihop_subset.py \
      --data=data\WebQA_train_val.json \
      --output=train_val_multihop.json
    ```
3. Create question subsets which the model predicted incorrectly

    ```bash
    python get_incorrect_subset.py \
      --preds=vlp/result/baseline_predictions_val.json \
      --output=vlp_incorrect_image_subset.json

    python get_incorrect_subset.py \
      --preds=text-retriever/data/WebQA_test_data/submission_image-based_val.json \
      --output=roberta_incorrect_image_subset.json
    ```

4. Create subsets containing common errors or unique errors of VLP and RoBERTa

    ```bash
    python get_common_unique_errors.py \
      --roberta=roberta_incorrect_image_subset.json \
      --vlp=vlp_incorrect_image_subset.json \
      --out-dir=.
    ```
