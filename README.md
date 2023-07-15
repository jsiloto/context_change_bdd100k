# Context Change for Split Computing in Autonomous Driving

## Current State
This repository currently is about object detection on BDD100k repository

## Environment


Download this repo and install the requirements


```bash
git clone git@github.com:jsiloto/context_change_bdd100k.git
cd context_change_bdd100k
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset
Download the dataset to `$DATASET`

```bash
DATASET=/data/datasets/bdd100k/
ls -l $DATASET
> bdd100k_images_100k.zip  bdd100k_labels_release.zip 
unzip $DATASET/bdd100k_images_100k.zip bdd100k/images/* -d $DATASET/..
unzip -j $DATASET/bdd100k_labels_release.zip bdd100k/labels/* -d $DATASET/labels_bdd100k
```

### Prepare Annotations
Annotations need to be prepared on the YOLO formate [Link Here]()
For that we use the approach of converting first to the coco format (BDD100k->COCO)
```bash
python scripts/bdd2coco.py '--bdd_dir' $DATASET
```
Now we convert the coco format annotations to the yolo format (COCO->YOLO)
```bash
python scripts/coco2yolo.py '--bdd_dir' $DATASET
```

# Training

Download the YOLOV5 fork and set up the environment

```bash
git clone git@github.com:jsiloto/yolov5.git
cd yolov5
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
export CUDA_VISIBLE_DEVICES=1
python train.py --img 640 --epochs 3 --data bdd100k.yaml --weights yolov5s.pt
```

# Other Dataset Specs

## Domain specific datasets
generate the domain specific datasets:
```bash
python scripts/create_domain_datasets.py '--bdd_dir' $DATASET
```

## Movie Dataset
```bash
DATASET=/data/datasets/bdd100kvideo/
ls -l $DATASET
> bdd100k_videos_train_14.zip  bdd100k_videos_val_04.zip
unzip -j $DATASET/bdd100k_videos_train_14.zip bdd100k/videos/* -d $DATASET/video/train
unzip -j $DATASET/bdd100k_videos_val_04.zip bdd100k/videos/* -d $DATASET/video/val
```
