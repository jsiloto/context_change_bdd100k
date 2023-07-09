import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Remove Unlabeled Images')
parser.add_argument('--bdd_dir', type=str, required=True)

def main(bdd_dir):
    labels_dir = os.path.join(bdd_dir, "labels/100k/val")
    images_dir = os.path.join(bdd_dir, "images/100k/val")

    labels = os.listdir(labels_dir)
    images = os.listdir(images_dir)
    labels = [i.split(".txt")[0] for i in labels]
    images = [i.split(".jpg")[0] for i in images]

    for i in images:
        if i not in labels:
            print(f"removing image {i}.jpg")
            path = os.path.join(images_dir, i) + ".jpg"
            os.remove(path)

if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg.bdd_dir)