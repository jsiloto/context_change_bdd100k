#!/bin/bash

python download_weights.py
python yolov5/val.py --data bdd100kvideo_train.yaml --weights weights/yolov5l_baseline.pt \
 --save-txt --project movie_dataset --name val
python yolov5/val.py --data bdd100kvideo_train.yaml --weights weights/yolov5l_baseline.pt \
 --save-txt --project movie_dataset --name train

mkdir -p /data/datasets/bdd100kvideo/labels/val
mv yolov5/movie_dataset/val/labels/* /data/datasets/bdd100kvideo/labels

mkdir -p /data/datasets/bdd100kvideo/labels/train
mv yolov5/movie_dataset/train/labels/* /data/datasets/bdd100kvideo/labels