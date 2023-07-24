#!/bin/bash

PARAMETERS_BASE="--img 640"

for data in "baseline" "clear" "daytime" "night" "partly_cloudy" \
  "residential" "city_street" "dawn_dusk" "highway" "overcast" "rainy" "snowy"; do

  for P in "02" "04" "06" "10"; do

    python yolov5/val.py --data yolov5/data/domains/${data}.yaml ${PARAMETERS_BASE} \
      --weights weights/all_yolov5s_split_${P}.pt \
      --project bdd_dataset/split_domain/ --name all_yolov5s_split_${P}.${data} 2>&1

    python yolov5/val.py --data yolov5/data/domains/${data}.yaml ${PARAMETERS_BASE} \
      --weights weights/vehicle_yolov5s_split_${P}.pt \
      --project bdd_dataset/split_domain/ --name vehicle_yolov5s_split_${P}.${data} 2>&1

  done
done
