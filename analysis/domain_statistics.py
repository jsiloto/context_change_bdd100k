import argparse
import copy
import os
from math import log2

import pandas as pd

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, required=True)
parser.add_argument('--split', type=str, default="val")

domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

def convert2percentage(d):
    dd = copy.copy(d)
    total_instances = 0
    for k, v in d.items():
        total_instances += v

    for k, v in d.items():
        dd[k] = round(1.0*v/total_instances, 4)

    return dd, total_instances

def get_distributions(cfg):
    class_distribution_dict = {}

    for d in domains:
        class_num = {}
        for i in range(9):
            class_num[i] = 0

        domain_dir = os.path.join(cfg.bdd_dir, "domains", d, "labels", cfg.split)
        labels = os.listdir(domain_dir)
        for l in labels:
            with open(os.path.join(domain_dir, l), "r") as fp:
                for line in fp:
                    c = int(line.split(" ")[0])
                    class_num[c] += 1


        class_distribution_dict[d] = convert2percentage(class_num)

    return class_distribution_dict

def h2(prob_dict):
    probs = [v for k,v in prob_dict.items()]
    c = [p*log2(p) for p in probs]
    return -sum(c)

def main(cfg):
    class_distribution_dict = get_distributions(cfg)
    for k, v in class_distribution_dict.items():
        print(f"{k} & {v[1]} & {round(h2(v[0]), 3)} \\\\")

    # df = pd.DataFrame(columns=domains, index=domains)
    # series = {}
    # df.loc[model] = pd.Series(series)

if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg)
