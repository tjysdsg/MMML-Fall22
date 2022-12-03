# OFA-retriever

#### Model Explanation

The basic idea of OFA retriever is to use a **Cross-Encoder** style retriever. Since the OFA is a unified structure, it directly concatenates the image token and text token into sequnence and generate the final results.

Concretely, for image-based questions, one example for OFA retriever is:

```python
Input: IMAGE_PATCHES + Is image caption " [IMAGE CAPTION] " related to the question of " [QUESTION] "?

Output: Is image caption " [IMAGE CAPTION] " related to the question of " [QUESTION] "? Yes (No)
```

For text-based questions, one example for OFA retriever is:

```python
Input: IMAGE_PATCHES + Is text " [TEXT FACT] " related to the question of " [QUESTION] "?

Output: Is text " [TEXT FACT] " related to the question of " [QUESTION] "? Yes (No)
```

In the text-based questions, the IMAGE_PATCHES are fully masked since we have no related pictures.



#### Run

**For the cached data:**

Directly place the downloaded data in the cache directory that is occupied by dummy_file/

The complete cache file can be downloaded through this link https://drive.google.com/file/d/1OvwrsFM-U2p0dxkhq4HvOLb1R3zKGKJt/view?usp=share_link

**For the OFA model:**

We need to follow the instructions in the huggingface documents and pip install transformers based on their suggestions in order to have access to the OFA model.

OFA-tiny: https://huggingface.co/OFA-Sys/ofa-tiny

OFA-medium: https://huggingface.co/OFA-Sys/ofa-medium

OFA-large: https://huggingface.co/OFA-Sys/ofa-large

**For training / testing / submission:**

```bash
# command for doing training on WebQA dataset 
sh webqa_train.sh
# command for doing inference on the current test dataset of WebQA
sh webqa_test.sh
# command for submmitting to the webqa leaderboard for final evaluation
sh webqa_submit.sh 
```



**TIPS:**

When applying this repo to your own version, you need to modify:

1. wandb API key
2. model type and model directory