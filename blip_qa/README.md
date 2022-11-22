Adapted from https://github.com/salesforce/BLIP

# Prerequisites

- Download BartScore pretrained model to `bart_score.pth`
- Modify paths in `configs/webqa.yaml`

# Train

```bash
python train_webqa.py
python train_webqa.py --resume output/WebQA/checkpoint09.pth
```

# Inference

```bash
python train_webqa.py --inference --resume output/WebQA/checkpoint09.pth
```
