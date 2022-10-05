# 1. Decode Images

```bash
python utils/decode_base64_images.py
```

# 2. Create image subset

```bash
python pos_neg_image_fact_analysis/subset_pos_neg_image_facts.py
```

Resulting `train.tsv` and `test.tsv`, they are already included in the repository

# 3. Run Positive/Negative Classifier

Training batch loss

![img](results/pos_neg_classification/training-batch-loss.png)

Test epoch accuracy

![img](results/pos_neg_classification/test-acc.png)
