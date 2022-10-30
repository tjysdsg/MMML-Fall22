#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../

ckpt_step=-1
if [[ "$1" = "" ]]; then
  echo "run $0 with checkpoint step (epoch) that you want to load from, for example $0 3"
  exit 1
else
  ckpt_step="$1"
fi

split="test"
output_dir=exp/retrieval_${split}
ckpts_dir=exp/retrieval/ckpt  # load from trained model

webqa_dir=/ocean/projects/cis210027p/shared/corpora/webqa
rcnn_feat=$webqa_dir
imgid_map=$webqa_dir/image_id_map_0328.pkl
detectron_dir=$webqa_dir/baseline_finetuned/detectron_weights

dataset=../subWebqa/data/${split}_subWebqa.json

python run_webqa.py \
  --do_predict \
  --split ${split} \
  --answer_provided_by 'img' \
  --task_to_learn 'filter' \
  --use_x_distractors \
  --amp \
  --new_segment_ids \
  --train_batch_size 1 \
  --num_workers 4 \
  --max_pred 10 --mask_prob 1.0 \
  --output_dir ${output_dir} \
  --ckpts_dir ${ckpts_dir} \
  --recover_step ${ckpt_step} \
  --txt_dataset_json_path $dataset \
  --img_dataset_json_path $dataset \
  --feature_folder $rcnn_feat \
  --detectron_dir $detectron_dir \
  --image_id_map_path $imgid_map

# Or you can specify the exact model path
# model_ckpt=$webqa_dir/baseline_finetuned/retrieval_x101fpn/model.3.bin
# python run_webqa.py \
#   --do_predict \
#   --split ${split} \
#   --answer_provided_by 'img' \
#   --task_to_learn 'filter' \
#   --use_x_distractors \
#   --amp \
#   --new_segment_ids \
#   --train_batch_size 1 \
#   --num_workers 4 \
#   --max_pred 10 --mask_prob 1.0 \
#   --output_dir ${output_dir} \
#   --ckpts_dir ${ckpts_dir} \
#   --txt_dataset_json_path $dataset \
#   --img_dataset_json_path $dataset \
#   --feature_folder $rcnn_feat \
#   --detectron_dir $detectron_dir \
#   --model_recover_path $model_ckpt \
#   --image_id_map_path $imgid_map
