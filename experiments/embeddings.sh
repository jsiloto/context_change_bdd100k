#!/bin/bash

for data in "baseline" "clear" "daytime" "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

  python yolov5/embeddings.py --weights weights/yolov5l_full.pt \
   --img-size 640 --data yolov5/data/domains/${data}.yaml \
   --project embeddings --name  --exist-ok

done