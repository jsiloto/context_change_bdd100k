#!/bin/bash

python download_weights.py

for M in "yolov5s.pt" "yolov5n.pt" "yolov5l.pt" "yolov5m.pt"; do
  for C in "all" "vehicle"; do
    weight=${C}_${M}
    rm -rf movie_dataset/${weight}

    python yolov5/val.py --data bdd100kvideo.yaml \
      --weights weights/${weight} --project movie_dataset \
      --name ${weight}/val --exist-ok --max-det 100 --save-txt --save-conf

    python yolov5/val.py --data bdd100kvideo_train.yaml \
      --weights weights/${weight} --project movie_dataset \
      --name ${weight}/train --exist-ok --max-det 100 --save-txt --save-conf
  done
done
