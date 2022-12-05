#!/usr/bin/env bash

if [[ "$1" = "" ]]; then
  echo "$0: Run with the submission file path"
  exit 1
fi

if [[ ! -f $1 ]]; then
  echo "$0: Cannot find $1"
  exit 1
fi

python submit_prediction.py $1
