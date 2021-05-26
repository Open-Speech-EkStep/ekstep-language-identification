#!/usr/bin/env bash

echo $(pwd)
train_set=$1
validation_set=$2
config_path=$3
mode=$4
echo $train_set
echo $validation_set
echo $config_path


#sleep 1000

if [ $mode = "train" ]; then
  echo "The mode is training...."
  #create manifests
  python3 data/create_manifest.py $train_set $validation_set
  #Start model train
  python3 model_training.py $config_path
elif [ $mode = "validation" ]; then
  echo "The mode is validation...."
  python3 model_validation.py $config_path
elif [ $mode = "acceptance" ]; then
  echo "The mode is acceptance...."
  python3 model_acceptance.py $config_path
elif [ $mode = "staging" ]; then
  echo "The mode is promote to staging...."
else
  echo "Invalid mode..."
fi
