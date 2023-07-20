import argparse
import copy
import os

import pandas as pd

import sys
sys.path.insert(0,'./yolov5')
from val_txt import txtval



parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--txt', type=str, default="val")

domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

def main(cfg):
    df = pd.DataFrame(columns=domains, index=domains)
    for d1 in domains:
        series = {}
        for d2 in domains:
            txtdir1 = os.path.join(cfg.txt, d1, "val", "labels")
            txtdir2 = os.path.join(cfg.txt, d2, "val", "labels")
            map1, map2, max_map = txtval(cfg.data, txtdir1, txtdir2)
            print(map1, map2, max_map)
            series[d2] = max_map

        df.loc[d1] = pd.Series(series)

    df = df.style.format(decimal='.', precision=3)
    print(df.to_latex())
    with open('mytable.tex', 'w') as tf:
        tf.write(df.to_latex())


if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg)
