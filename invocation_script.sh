#!/usr/bin/env bash

echo $(pwd)
train_set=$1
validation_set=$2
tracking_ui=$3
mode=$4
echo $train_set
echo $validation_set
echo $tracking_ui


#sleep 1000

if [ $mode = "train" ]; then
  echo "The mode is training...."
  #create manifests
  python3 data/create_manifest.py $train_set $validation_set
  #Start model train
  python3 train.py $tracking_ui
elif [ $mode = "validation" ]; then
  echo "The mode is validation...."
  python3 batch_validation.py $tracking_ui $validation_set
elif [ $mode = "acceptance" ]; then
  echo "The mode is acceptance...."
  python3 model_acceptance.py $tracking_ui .90
elif [ $mode = "staging" ]; then
  echo "The mode is promote to staging...."
else
  echo "Invalid mode..."
fi
