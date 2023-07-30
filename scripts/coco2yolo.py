# Adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/converter.py
import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser(description='coco2yolo')
parser.add_argument('--config', type=str, required=True)


def coco91_to_coco80_class(c):  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]

    return x[c]

def convert_coco(dataset_specs):
    """Converts COCO dataset annotations to a format suitable for training YOLOv5 models.
    Args:
        dataset_specs (dict): dataset specifications with input and output directories
    """
    base_dir = dataset_specs['path']
    coco_train_annotations = os.path.join(base_dir, dataset_specs['coco']['annotations']['train'])
    coco_val_annotations = os.path.join(base_dir, dataset_specs['coco']['annotations']['val'])
    if 'labels' in dataset_specs:
        yolo_labels_dir = os.path.join(base_dir, dataset_specs['labels'])
    else:
        yolo_labels_dir = os.path.join(base_dir, "labels/")

    shutil.rmtree(yolo_labels_dir, ignore_errors=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir + "/train", exist_ok=True)
    os.makedirs(yolo_labels_dir + "/val", exist_ok=True)

    ignored_classes = []
    if 'ignored_classes' in dataset_specs:
        ignored_classes = dataset_specs['ignored_classes']

    def convert_coco_annotation(annotation_file, save_dir):
        with open(annotation_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {annotation_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                cls = coco91_to_coco80_class(ann['category_id'] - 1)  # class
                if ann['iscrowd']:
                    continue
                if cls in ignored_classes:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((save_dir / f).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    line = *(bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')

    convert_coco_annotation(coco_train_annotations, Path(yolo_labels_dir + "/train"))
    convert_coco_annotation(coco_val_annotations, Path(yolo_labels_dir + "/val"))


def rle2polygon(segmentation):
    """
    Convert Run-Length Encoding (RLE) mask to polygon coordinates.

    Args:
        segmentation (dict, list): RLE mask representation of the object segmentation.

    Returns:
        (list): A list of lists representing the polygon coordinates for each contour.

    Note:
        Requires the 'pycocotools' package to be installed.
    """
    from pycocotools import mask

    m = mask.decode(segmentation)
    m[m > 0] = 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = contour_approx.flatten().tolist()
        polygons.append(polygon)
    return polygons


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.array): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.array): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


if __name__ == '__main__':
    cfg = parser.parse_args()
    dataset_specs = yaml.safe_load(open(cfg.config))
    convert_coco(dataset_specs)
