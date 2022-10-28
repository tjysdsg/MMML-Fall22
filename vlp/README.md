# Prerequisite

## nvidia apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex

# somehow the newest version doesn't work on pytorch 1.8.1+cuda10.2
git checkout --hard f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Files

- RCNN features, either downloaded from https://github.com/WebQnA/WebQA_Baseline, or extracted by yourself
- Files in http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_data_first_release.7z
    - WebQA json files (`WebQA_train_val.json` and `WebQA_test.json`)
    - Image id map (`image_id_map_0328.pkl`)
- Pretrained detectron fc7 weights and biases (`fc7_w.pkl` and `fc7_w.pkl`),
  see https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz
- Pretrained WebQA VLP model https://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_baseline_ckpts.7z

# Retrieval Baseline

## Training

- Text-only questions are removed
- RCNN features + image caption's text embeddings

**Read the script, you might want to change arguments**

```bash
./train_retrieval.sh
./infer_retrieval.sh <checkpoint_number>  # infer on train/val set
```

## Inference on Test Set

TODO

# Question Answering

TODO
