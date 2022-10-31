#!/usr/bin/env bash

if [[ "$1" = "" ]]; then
  echo "$0: Run with the submission file path"
  exit 1
fi

if [[ ! -f $1 ]]; then
  echo "$0: Cannot find $1"
  exit 1
fi

webqa_dir=/ocean/projects/cis210027p/shared/corpora/webqa
imgid_map=$webqa_dir/image_id_map_0328.pkl

mkdir -p tmp
python fix_submission_image_id.py --imgid-map=$imgid_map $1 tmp/submission.json || exit 1
python submit_prediction.py tmp/submission.json
