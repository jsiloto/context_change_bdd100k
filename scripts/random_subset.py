import os
import json
import argparse
import random

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Remove Unlabeled Images')
parser.add_argument('--bdd_dir', type=str, required=True)
parser.add_argument('--num', type=int, required=True)


def main(bdd_dir, num):
    labels_dir = os.path.join(bdd_dir, f"labels/100k/train")
    images_dir = os.path.join(bdd_dir, f"images/100k/train")

    labels = os.listdir(labels_dir)
    images = os.listdir(images_dir)
    labels = [i.split(".txt")[0] for i in labels]
    images = [i.split(".jpg")[0] for i in images]

    final_images = random.sample(images, num)

    for i in images:
        if i not in final_images:
            print(f"removing image {i}.jpg")
            path = os.path.join(images_dir, i) + ".jpg"
            os.remove(path)
            path = os.path.join(labels_dir, i) + ".txt"
            os.remove(path)


if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg.bdd_dir, cfg.num)
