#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../

webqa_dir=/ocean/projects/cis210027p/shared/corpora/webqa
dataset=data/webqa_subset.json
rcnn_feat=$webqa_dir
imgid_map=$webqa_dir/image_id_map_0328.pkl
detectron_dir=$webqa_dir/baseline_finetuned/detectron_weights

output_dir=exp/retrieval
ckpts_dir=${output_dir}/ckpt

ckpt_step=-1
if [[ "$1" = "" ]]; then
  echo "run $0 with checkpoint step (epoch) that you want to load from, for example $0 3"
  exit 1
else
  ckpt_step="$1"
fi

python run_webqa.py \
  --do_predict \
  --split "train|val" \
  --answer_provided_by 'img' \
  --task_to_learn 'filter' \
  --use_x_distractors \
  --amp \
  --new_segment_ids \
  --train_batch_size 1 \
  --num_workers 4 \
  --max_pred 10 --mask_prob 1.0 \
  --learning_rate 3e-5 \
  --save_loss_curve \
  --num_train_epochs 6 \
  --output_dir ${output_dir}_infer \
  --ckpts_dir ${ckpts_dir} \
  --txt_dataset_json_path $dataset \
  --img_dataset_json_path $dataset \
  --feature_folder $rcnn_feat \
  --detectron_dir $detectron_dir \
  --image_id_map_path $imgid_map \
  --recover_step $ckpt_step
