## https://gist.github.com/zzzxxxttt/4ea0e1bac0293fee518035ae52394746

import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, required=True)
parser.add_argument('--scene', type=str, default="")
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels_bdd100k', 'bdd100k_labels_images_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels_bdd100k', 'bdd100k_labels_images_train.json')

os.makedirs(os.path.join(cfg.bdd_dir, 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_val_coco.json')
dst_train_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_train_coco.json')

scene = ["city street", "tunnel", "highway", "residential", "parking lot", "gas stations", "undefined"]
if len(cfg.scene) > 0:
  scene = [cfg.scene]



def bdd2coco_detection(labeled_images, save_dir):
  attr_dict = {"categories":
    [
      {"supercategory": "none", "id": 1, "name": "person"},
      {"supercategory": "none", "id": 2, "name": "car"},
      {"supercategory": "none", "id": 3, "name": "rider"},
      {"supercategory": "none", "id": 4, "name": "bus"},
      {"supercategory": "none", "id": 5, "name": "truck"},
      {"supercategory": "none", "id": 6, "name": "bike"},
      {"supercategory": "none", "id": 7, "name": "motor"},
      {"supercategory": "none", "id": 8, "name": "traffic light"},
      {"supercategory": "none", "id": 9, "name": "traffic sign"},
      # {"supercategory": "none", "id": 10, "name": "train"},
    ]}

  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

  images = list()
  annotations = list()
  ignore_categories = set()

  counter = 0
  for i in tqdm(labeled_images):
    if i["attributes"]['scene'] not in scene:
      continue

    counter += 1
    image = dict()
    image['file_name'] = i['name']
    image['height'] = 720
    image['width'] = 1280
    image['attributes'] = i['attributes']

    image['id'] = counter

    empty_image = True

    tmp = 0
    for l in i['labels']:
      annotation = dict()
      if l['category'] in id_dict.keys():
        tmp = 1
        empty_image = False
        annotation["iscrowd"] = 0
        annotation["image_id"] = image['id']
        x1 = l['box2d']['x1']
        y1 = l['box2d']['y1']
        x2 = l['box2d']['x2']
        y2 = l['box2d']['y2']
        annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        annotation['area'] = float((x2 - x1) * (y2 - y1))
        annotation['category_id'] = id_dict[l['category']]
        annotation['ignore'] = 0
        annotation['id'] = l['id']
        annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        annotations.append(annotation)
      else:
        ignore_categories.add(l['category'])

    if empty_image:
      print('empty image!')
      continue
    if tmp == 1:
      images.append(image)

  attr_dict["images"] = images
  attr_dict["annotations"] = annotations
  attr_dict["type"] = "instances"

  print('ignored categories: ', ignore_categories)
  print('saving...')
  with open(save_dir, "w") as file:
    json.dump(attr_dict, file, indent=2)
  print('Done.')


def main():
  # create BDD training set detections in COCO format
  print('Loading training set...')
  with open(src_train_dir) as f:
    train_labels = json.load(f)
  print('Converting training set...')
  bdd2coco_detection(train_labels, dst_train_dir)

  # create BDD validation set detections in COCO format
  print('Loading validation set...')
  with open(src_val_dir) as f:
    val_labels = json.load(f)
  print('Converting validation set...')
  bdd2coco_detection(val_labels, dst_val_dir)


if __name__ == '__main__':
  main()
