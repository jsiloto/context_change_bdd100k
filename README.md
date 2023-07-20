# Context Change for Split Computing in Autonomous Driving

## Current State
This repository currently is about object detection on BDD100k repository

## Environment

Download this repo, install the requirements, download the YOLOV5 fork
```bash
git clone git@github.com:jsiloto/context_change_bdd100k.git
cd context_change_bdd100k
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
git clone git@github.com:jsiloto/yolov5.git
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

# Other Dataset Specs

## Domain specific datasets
generate the domain specific datasets:
```bash
python scripts/create_domain_datasets.py '--bdd_dir' $DATASET
```

## Movie Dataset

### Create the frame by frame dataset
First be sure to download the appropriate files for reproduction
and unzip to the correct location
```bash
DATASET=/data/datasets/bdd100kvideo/
ls -l $DATASET
> bdd100k_videos_train_14.zip  bdd100k_videos_val_04.zip
unzip -j $DATASET/bdd100k_videos_train_14.zip bdd100k/videos/* -d $DATASET/video/train
unzip -j $DATASET/bdd100k_videos_val_04.zip bdd100k/videos/* -d $DATASET/video/val
```

Run the following scripts which will create the frame by frame dataset,
annotate it using the yolov5 format and evaluate on domain specific models
```bash
./scripts/movie_dataset_build.sh
./scripts/movie_dataset_domain.sh
```

Now, we have built and evaluated the movie dataset with many model, we can
see which models are better together

```bash
python scripts/domain_switching_map.py
```


```bash
export CUDA_VISIBLE_DEVICES=1
python yolov5/train.py --img 640 --epochs 3 --data bdd100k.yaml --weights yolov5s.pt
```