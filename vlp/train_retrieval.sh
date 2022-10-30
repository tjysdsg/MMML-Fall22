#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../

output_dir=exp/retrieval
ckpts_dir=${output_dir}/ckpt

webqa_dir=/ocean/projects/cis210027p/shared/corpora/webqa
rcnn_feat=$webqa_dir/features
imgid_map=$webqa_dir/image_id_map_0328.pkl
detectron_dir=$webqa_dir/baseline_finetuned/detectron_weights

# model_ckpt=$webqa_dir/baseline_finetuned/retrieval_x101fpn/model.3.bin
# train from vlp that has not been finetuned on WebQA
model_ckpt=$webqa_dir/vlp_pretrained/model.30.bin

dataset=../subWebqa/data/train_subWebqa.json
# dataset=data/webqa_subset.json

python run_webqa.py \
  --do_train \
  --split train \
  --answer_provided_by 'img' \
  --task_to_learn 'filter' \
  --use_x_distractors \
  --amp \
  --new_segment_ids \
  --train_batch_size 1 \
  --num_workers 4 \
  --max_pred 10 --mask_prob 1.0 \
  --learning_rate 3e-5 \
  --gradient_accumulation_steps 128 \
  --save_loss_curve \
  --num_train_epochs 6 \
  --output_dir ${output_dir} \
  --ckpts_dir ${ckpts_dir} \
  --txt_dataset_json_path $dataset \
  --img_dataset_json_path $dataset \
  --feature_folder $rcnn_feat \
  --model_recover_path $model_ckpt \
  --detectron_dir $detectron_dir \
  --image_id_map_path $imgid_map
