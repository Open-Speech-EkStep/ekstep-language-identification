#!/usr/bin/env bash

echo `pwd`
echo $1
echo $2
echo $3
#create manifests
python3 data/create_manifest.py $1 $2
#Start model train
nohup python3 train.py $3 &
#sleep 1000
