## 0. `subData/`
Generated from [image objects selected by TJY](https://github.com/tjysdsg/MMML-Fall22/blob/main/webqa/pos_neg_image_fact_analysis/train.tsv), but integrated necessary label information. Find preparation code in `sbert.ipynb`.

## 1. `sbertFeats/`
Extracted using [sentence-transformers](https://github.com/UKPLab/sentence-transformers). Features stored in `sbertFeats` (17MB). Find preparation code in `sbert.ipynb`

## 2. `figs/`
use following command to generate image caption-based pca visualization
```
python plot.py -h
```
Detail Usage
```
usage: plot.py [-h] [--feats-dir FEATS_DIR] [--task {posneg,qcate}] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  --feats-dir FEATS_DIR
  --task {posneg,qcate}
  --out OUT, -o OUT
```
Find output figs in `figs/pca`