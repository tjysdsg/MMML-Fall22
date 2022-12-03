Adapted from https://github.com/salesforce/BLIP

# Prerequisites

- Download BartScore pretrained model
  to `bart_score.pth`: https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing
  (Source: https://github.com/neulab/BARTScore)
- Modify paths in `configs/webqa.yaml`
    - Reduce batch size if needed, batch of 4 takes about 22GB of VRAM

# Train

**Modify wandb API Key in `train_webqa.py`**

```python
def init_wandb(output_dir: str):
    # need to change to your own API when using
    os.environ['EXP_NUM'] = 'WebQA'
    os.environ['WANDB_NAME'] = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(int(round(time.time() * 1000)) / 1000)
    )
    os.environ['WANDB_API_KEY'] = 'b6bb57b85f5b5386441e06a96b564c28e96d0733'  # <----
    os.environ['WANDB_DIR'] = output_dir
    wandb.init(project="blip_webqa_qa_img_only")
```

**Train**

```bash
python train_webqa.py
python train_webqa.py --resume output/WebQA/checkpoint09.pth
```

# Inference

```bash
python train_webqa.py --inference --resume output/WebQA/checkpoint09.pth
# output/WebQA/*_pred.json
```
