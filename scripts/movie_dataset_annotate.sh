#!/bin/bash

python download_weights.py
python yolov5/val.py --data bdd100kvideo.yaml --weights weights/yolov5l_baseline.pt \
 --save-txt --project movie_dataset --name val --exist-ok
python yolov5/val.py --data bdd100kvideo_train.yaml --weights weights/yolov5l_baseline.pt \
 --save-txt --project movie_dataset --name train --exist-ok

mkdir -p /data/datasets/bdd100kvideo/labels/val
cp movie_dataset/val/labels/* /data/datasets/bdd100kvideo/labels/val

mkdir -p /data/datasets/bdd100kvideo/labels/train
cp movie_dataset/train/labels/* /data/datasets/bdd100kvideo/labels/train