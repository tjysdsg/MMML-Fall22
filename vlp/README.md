# Prerequisite

## nvidia apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex

# somehow the newest version doesn't work on pytorch 1.8.1+cuda10.2
git checkout --hard f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Train Retrieval Baseline

- Text+image

```bash
./train_retrieval.sh
```