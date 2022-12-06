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

On validation:

```bash
python train_webqa.py --inference --resume output/WebQA/checkpoint09.pth
# output/WebQA/*_pred.json
```

On test:

```bash
python filter_test_facts_from_retrieval_pred.py --retrieval-results=xxx # creates test.json
python train_webqa.py --inference --inference_split=test --resume=xxx.pth
python generate_submission.py
```

# Results

## Image-based questions

Beam size = 10

Validation set with only image-based questions:

```json
{
  "color": 0.5415536046820965,
  "shape": 0.26126126126126126,
  "YesNo": 0.46215780998389694,
  "number": 0.3894649751792609,
  "text": 0,
  "Others": 0.7206404752328825,
  "choose": 0.7317241835044884,
  "f1": 0.44761904761904775,
  "recall": 0.7253919880871498,
  "acc": 0.5771579218875255,
  "fl": 0.39096362071974833,
  "qa": 0.22564775086823263
}
```