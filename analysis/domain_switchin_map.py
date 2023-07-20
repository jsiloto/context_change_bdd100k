import argparse
import copy
import os
from math import log2

import pandas as pd

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, required=True)
parser.add_argument('--txt_dir', type=str, default="val")

domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

def main(cfg):
    for d1 in domains:
        for d2 in domains:
            txtdir1 = os.path.join(cfg.txt_dir, d1, "labels")
            txtdir2 = os.path.join(cfg.txt_dir, d2, "labels")
            map1, map2, max_map = txtval(cfg.bdd_dir, txtdir1, txtdir2)

if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg)
