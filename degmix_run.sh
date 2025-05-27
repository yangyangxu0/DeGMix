#!/bin/bash
CONFIG=$1
GPU=$2
"python ./src/main.py --cfg $CONFIG --datamodule.data_dir ../../datasets/ --trainer.gpus $GPU"
cat $CONFIG
