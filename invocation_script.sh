#!/usr/bin/env bash

#echo `pwd`
#echo $1
#echo $2
#create manifests
python data/create_manifest.py $1 $2
#Start model train
python train.py
#sleep 1000
