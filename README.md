# Context Change for Split Computing in Autonomous Driving

## Current State
This repository currently is about object detection on BDD100k repository

## Environment
Install the requirements


```bash
pip install -r requirements.txt
```

### Dataset
Download the dataset to `$DATASET`

```bash
DATASET=/data/datasets/bdd100k/
ls -l $DATASET
> bdd100k_images_100k.zip  bdd100k_labels_release.zip 
unzip -j $DATASET/bdd100k_images_100k.zip bdd100k/images/* -d $DATASET/images
unzip -j $DATASET/bdd100k_labels_release.zip bdd100k/labels/* -d $DATASET/labels
```

### Prepare Annotations
Annotations need to be prepared on the YOLO formate [Link Here]()
For that we use the approach of converting first to the coco format (BDD100k->COCO)
```bash
python scripts/bdd2coco.py '--bdd_dir' $DATASET
```
Now we convert the coco format annotations to the yolo format (COCO->YOLO)
```bash
convert
```