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

## Evaluate on Test Data

This will print evaluation result returned by WebQA evaluation server

```bash
./submit_test.sh test_predictions.json
```

# Question Answering

TODO

# Results

## Retrieval

### [Official baseline](https://github.com/WebQnA/WebQA_Baseline) on validation set, excluding Qcate='text'

```
th = 0.05
   pr.mean = 0.5590773799673476
   re.mean = 0.9101005702741697
   f1.mean = 0.6445277373537162
   
th = 0.1
   pr.mean = 0.6202838370065701
   re.mean = 0.861840766475728
   f1.mean = 0.6789399043602102
   
th = 0.15
   pr.mean = 0.6413196194191658
   re.mean = 0.8193804344660799
   f1.mean = 0.6820918119017902
   
th = 0.2
   pr.mean = 0.652028305766226
   re.mean = 0.78479080868834
   f1.mean = 0.6787558641121125
   
th = 0.25
   pr.mean = 0.6572354824313864
   re.mean = 0.750408286236492
   f1.mean = 0.6701178575473179
   
th = 0.3
   pr.mean = 0.6490975369783404
   re.mean = 0.7168542683470219
   f1.mean = 0.6536161807344882
   
th = 0.35
   pr.mean = 0.6430776911327043
   re.mean = 0.6859928612483861
   f1.mean = 0.6379211370917323
   
th = 0.4
   pr.mean = 0.6285327775488734
   re.mean = 0.6526459736476383
   f1.mean = 0.6164943935672113
   
th = 0.45
   pr.mean = 0.6112487427674252
   re.mean = 0.6188848358397654
   f1.mean = 0.5934317727953615
   
th = 0.5
   pr.mean = 0.5866927208491565
   re.mean = 0.5830524656628416
   f1.mean = 0.5656329718693881
   
th = 0.55
   pr.mean = 0.5630411477859789
   re.mean = 0.5480486062704875
   f1.mean = 0.5376836015243321
   
th = 0.6
   pr.mean = 0.542022019982733
   re.mean = 0.5180157016601658
   f1.mean = 0.5129939458690207
   
th = 0.65
   pr.mean = 0.5053710222997345
   re.mean = 0.4761767667388758
   f1.mean = 0.47551170627909845
   
th = 0.7
   pr.mean = 0.4716720136180335
   re.mean = 0.4351663260096445
   f1.mean = 0.43969728040887784
   
th = 0.75
   pr.mean = 0.4352942090774728
   re.mean = 0.3912561649414954
   f1.mean = 0.40073559351740734
   
th = 0.8
   pr.mean = 0.397804829331351
   re.mean = 0.3477602374879605
   f1.mean = 0.3602638689009434
   
th = 0.85
   pr.mean = 0.33918891956392355
   re.mean = 0.2926653913456048
   f1.mean = 0.3055973771431768
   
th = 0.9
   pr.mean = 0.25940105057296026
   re.mean = 0.2164439588612333
   f1.mean = 0.2296788151612831
   
th = 0.95
   pr.mean = 0.12151223402578375
   re.mean = 0.10066195826147249
   f1.mean = 0.10734452239323374
```
