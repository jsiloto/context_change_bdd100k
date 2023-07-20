#!/bin/bash

python download_weights.py


## Create Dataset Files
rm -rf /data/datasets/bdd100kvideo/images
python scripts/movie_dataset.py --movie_dir /data/datasets/bdd100kvideo/ --num_clips 25 --split train
python scripts/movie_dataset.py --movie_dir /data/datasets/bdd100kvideo/ --num_clips 25 --split val


# Annotate Dataset
rm -rf /data/datasets/bdd100kvideo/labels
rm -rf movie_dataset/val
rm -rf movie_dataset/train

python yolov5/val.py --data bdd100kvideo.yaml --weights weights/yolov5l_full.pt \
 --save-txt --project movie_dataset --name val --exist-ok --max-det 100
python yolov5/val.py --data bdd100kvideo_train.yaml --weights weights/yolov5l_full.pt \
 --save-txt --project movie_dataset --name train --exist-ok --max-det 100

mkdir -p /data/datasets/bdd100kvideo/labels/val
cp movie_dataset/val/labels/* /data/datasets/bdd100kvideo/labels/val

mkdir -p /data/datasets/bdd100kvideo/labels/train
cp movie_dataset/train/labels/* /data/datasets/bdd100kvideo/labels/train