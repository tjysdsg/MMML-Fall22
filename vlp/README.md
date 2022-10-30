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
```

## Inference

```bash
./infer_retrieval.sh <checkpoint_number>  # works on train/val/test
```

# Question Answering

TODO

# Results

## Retrieval

### [Official baseline](https://github.com/WebQnA/WebQA_Baseline) on subWebqa/val

th = 0.05

- pr.mean = 0.5866060782495129
- re.mean = 0.9084212863125564
- f1.mean = 0.6657251532278755

th = 0.1

- pr.mean = 0.6488824355191198
- re.mean = 0.8639771910006059
- f1.mean = 0.700370300581172

th = 0.15

- pr.mean = 0.6692734094472224
- re.mean = 0.8264296205564478
- f1.mean = 0.7039630354364256

th = 0.2

- pr.mean = 0.6784837970388804
- re.mean = 0.7938628450664067
- f1.mean = 0.7004150556650198

th = 0.25

- pr.mean = 0.6783912414221015
- re.mean = 0.7632117572872118
- f1.mean = 0.6906237382541671

th = 0.3

- pr.mean = 0.665625330063575
- re.mean = 0.727579866880658
- f1.mean = 0.6705303103416815

th = 0.35

- pr.mean = 0.6527658072011224
- re.mean = 0.6988444898320341
- f1.mean = 0.6528317136782796

th = 0.4

- pr.mean = 0.6428552227458735
- re.mean = 0.6739404728586189
- f1.mean = 0.637816369670561

th = 0.45

- pr.mean = 0.6300583061587308
- re.mean = 0.6409905536878154
- f1.mean = 0.6176560500344097

th = 0.5

- pr.mean = 0.6044262961637928
- re.mean = 0.6076574869996286
- f1.mean = 0.5902307498729092

th = 0.55

- pr.mean = 0.5839155548148685
- re.mean = 0.5762401233687712
- f1.mean = 0.5659617644616928

th = 0.6

- pr.mean = 0.561131545022073
- re.mean = 0.5440564915594926
- f1.mean = 0.5395727739480264

th = 0.65

- pr.mean = 0.5205188558932922
- re.mean = 0.4996124346137504
- f1.mean = 0.498321743669181

th = 0.7

- pr.mean = 0.4842867105628339
- re.mean = 0.45746721673285823
- f1.mean = 0.4609641094089011

th = 0.75

- pr.mean = 0.4393318192027081
- re.mean = 0.4111074835861323
- f1.mean = 0.41687788913076407

th = 0.8

- pr.mean = 0.39654801405709367
- re.mean = 0.36628028295962745
- f1.mean = 0.37337911782136823

th = 0.85

- pr.mean = 0.3376724198631857
- re.mean = 0.3080432453374753
- f1.mean = 0.31583202853513404

th = 0.9

- pr.mean = 0.24495294877395776
- re.mean = 0.21685631540086533
- f1.mean = 0.22515641925892155

th = 0.95

- pr.mean = 0.1118763149241378
- re.mean = 0.09961596814151924
- f1.mean = 0.1032169044246162
