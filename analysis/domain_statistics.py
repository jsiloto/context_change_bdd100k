import argparse
import copy
import os

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

    return dd

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

        print(d)
        class_distribution_dict[d] = convert2percentage(class_num)

    return class_distribution_dict

def main(cfg):



    df = pd.DataFrame(columns=domains, index=domains)
    series = {}
    df.loc[model] = pd.Series(series)

if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg)






for model in domains:
    series = {}
    for data in domains:
        results_file = f"domains/{model}.{data}/results.json"
        with open(results_file, "r") as fp:
            results = json.load(fp)
            series[data] = results['map']

    df.loc[model] = pd.Series(series)