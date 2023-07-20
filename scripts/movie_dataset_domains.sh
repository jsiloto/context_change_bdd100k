#!/bin/bash

python download_weights.py

for D in "baseline" "clear" "daytime"  "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"
do
    rm -rf movie_dataset/${D}

    python yolov5/val.py --data bdd100kvideo.yaml \
     --weights weights/yolov5s_${D}.pt --project movie_dataset \
     --name ${D}/val --exist-ok --max-det 100 --save-txt --save-conf

    python yolov5/val.py --data bdd100kvideo_train.yaml \
     --weights weights/yolov5s_${D}.pt --project movie_dataset \
     --name ${D}/train --exist-ok --max-det 100 --save-txt --save-conf
done