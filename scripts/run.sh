#!/bin/bash

workdir="$(dirname "$0")"
cd ${workdir}/../

export PYTHONPATH=$PYTHONPATH:/home/qizheng/code/backbone_extract/models/yolov3/yolov3-master
export PYTHONPATH=$PYTHONPATH:/home/qizheng/code/backbone_extract/models/deeplabv3/pytorch-deeplab-xception-master
export PYTHONPATH=$PYTHONPATH:/home/qizheng/code/backbone_extract/models/cycle_gan/pytorch-CycleGAN-and-pix2pix-master

PY="venv/Scripts/python"

$PY analysize.py
