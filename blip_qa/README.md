Adapted from https://github.com/salesforce/BLIP

# Prerequisites

- Download BartScore pretrained model
  to `bart_score.pth`: https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing
  (Source: https://github.com/neulab/BARTScore)
- Modify paths in `configs/webqa.yaml`
    - Reduce batch size if needed, batch of 4 takes about 22GB of VRAM

# Train

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

### Validation set with only image-based questions but no image input

```json
{
  "color": 0.38289438680500126,
  "shape": 0.2545045045045045,
  "YesNo": 0.43156199677938806,
  "number": 0.38516271373414235,
  "Others": 0.7214058080612734,
  "choose": 0.7238565976248201,
  "f1": 0.4063148542999289,
  "recall": 0.722456445431812,
  "acc": 0.5537468746963591,
  "fl": 0.382265955848317,
  "qa": 0.23300764244162658
}
```

### Validation set with only image-based questions

```json
{
  "color": 0.5415536046820965,
  "shape": 0.26126126126126126,
  "YesNo": 0.46215780998389694,
  "number": 0.3894649751792609,
  "Others": 0.7206029891173931,
  "choose": 0.730939448242445,
  "f1": 0.44761904761904775,
  "recall": 0.7250341611761265,
  "acc": 0.5769910499987129,
  "fl": 0.39096361411710323,
  "qa": 0.24599834998807873
}
```

### Validation set with only image-based questions + multitask qcate prediction:

```json
{
  "color": 0.5713487629688748,
  "shape": 0.3063063063063063,
  "YesNo": 0.43317230273752017,
  "number": 0.39094502665931236,
  "Others": 0.7208597507757726,
  "choose": 0.7370198894908607,
  "f1": 0.43646233120113714,
  "recall": 0.7277874959807036,
  "acc": 0.5723212591011274,
  "fl": 0.4019844882602969,
  "qa": 0.2511664842523028
}
```

### Validation set with only image-based questions + multitask qcate prediction + MED

```json
{
  "color": 0.4980712955573291,
  "shape": 0.23198198198198194,
  "YesNo": 0.499597423510467,
  "number": 0.39703989703989695,
  "text": 0,
  "Others": 0.6899336823300184,
  "choose": 0.7012885376253227,
  "f1": 0.46479211087420025,
  "recall": 0.6948014341303971,
  "acc": 0.5720565145113999,
  "fl": 0.34005613169419086,
  "qa": 0.2227310447499811
}
```

### Test set, img_only_multitask model image-based outputs merged with T5 text-based outputs

FL is low because BLIP is uncased while the test is cased

```json
{
  "Retrieval": 0.7573,
  "QA-FL": 0.3829,
  "QA-Acc": 0.5179,
  "QA": 0.2743
}
```

# Case Study

```
"d5cd225c0dba11ecb1e81171463288e9"

Q: Is the base of Opening Doors sculpture at Eden Villa Park taller than the base of Chen Wenqin's Infinity Curve sculpture?
A: The base of Opening Doors sculpture at Eden Villa Park is shorter than the base of Chen Wenqin's Infinity Curve sculpture.
P: the base of opening doors sculpture at eden villa park is not taller than the base of che guang's olympic sculpture sculpture

bart_score: 0.13739829230954648
topic: "modern artwork"
split: "val"
Qcate: "YesNo"
```
