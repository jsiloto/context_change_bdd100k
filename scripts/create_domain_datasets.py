import os
import json
import argparse
import random
import shutil
import sys

import yaml




from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, required=True)
parser.add_argument('--nocopy', action='store_true')
cfg = parser.parse_args()


def main():
    domains = {
        "baseline": ["baseline"],
        "scene": ["residential", "highway", "city street"],
        "weather": ["overcast", "snowy", "rainy", "partly cloudy", "clear"],
        "timeofday": ["dawn/dusk", "daytime", "night"]
    }
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "bdd100k.yaml")
    with open(yaml_path, 'r') as file:
        datayaml = yaml.safe_load(file)



    src_val_dir = os.path.join(cfg.bdd_dir, 'labels_bdd100k', 'bdd100k_labels_images_val.json')
    src_train_dir = os.path.join(cfg.bdd_dir, 'labels_bdd100k', 'bdd100k_labels_images_train.json')

    print('Loading training set...')
    with open(src_train_dir) as f:
        train_labels = json.load(f)

    print('Loading validation set...')
    with open(src_val_dir) as f:
        val_labels = json.load(f)
    shutil.rmtree("./data", ignore_errors=True)
    os.makedirs("./data", exist_ok=True)

    for k, v in domains.items():
        for name in v:
            if name == "baseline":
                train_list = [l["name"].split(".jpg")[0] for l in train_labels]
                val_list = [l["name"].split(".jpg")[0] for l in val_labels]
            else:
                train_list = [l["name"].split(".jpg")[0] for l in train_labels if l['attributes'][k] == name]
                val_list = [l["name"].split(".jpg")[0] for l in val_labels if l['attributes'][k] == name]

            train_list = random.sample(train_list, 4800)

            if "/" in name:
                name = name.replace("/", "_")
            if " " in name:
                name = name.replace(" ", "_")

            domain_dir = os.path.join(cfg.bdd_dir, "domains", name)
            datayaml['path'] = domain_dir + "/images/train"
            datayaml['train'] = domain_dir + "/images/val"


            with open(f'./data/{name}.yaml', 'w+') as file:
                documents = yaml.dump(datayaml, file)

            if not cfg.nocopy:
                shutil.rmtree(domain_dir, ignore_errors=True)
                os.makedirs(domain_dir + "/images/train", exist_ok=True)
                os.makedirs(domain_dir + "/images/val", exist_ok=True)
                os.makedirs(domain_dir + "/labels/train", exist_ok=True)
                os.makedirs(domain_dir + "/labels/val", exist_ok=True)
                for fname in tqdm(train_list):
                    src = os.path.join(cfg.bdd_dir, "images/100k/train", f"{fname}.jpg")
                    dst = os.path.join(domain_dir, "images/train", f"{fname}.jpg")
                    shutil.copy(src, dst)
                    src = os.path.join(cfg.bdd_dir, "labels/100k/train", f"{fname}.txt")
                    dst = os.path.join(domain_dir, "labels/train", f"{fname}.txt")
                    shutil.copy(src, dst)

                for fname in tqdm(val_list):
                    src = os.path.join(cfg.bdd_dir, "images/100k/val", f"{fname}.jpg")
                    dst = os.path.join(domain_dir, "images/val", f"{fname}.jpg")
                    shutil.copy(src, dst)
                    src = os.path.join(cfg.bdd_dir, "labels/100k/val", f"{fname}.txt")
                    dst = os.path.join(domain_dir, "labels/val", f"{fname}.txt")
                    shutil.copy(src, dst)






if __name__ == '__main__':
    main()
