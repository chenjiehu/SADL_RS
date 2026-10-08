# SADL

Code for *Generalized Few-Shot Remote Sensing Segmentation Based on Strong Data Augmentation and Domain-Irrelevant Feature Learning*.

## Layout

```
SADL/
├── pretrain.py                  stage 1: source-domain pre-training
├── adapt.py                     stage 2: joint training on source + target shots, evaluation on target query
├── requirements.txt
├── tools/
│   └── split_shot_query.py      generate shot / query name lists
└── sadl/
    ├── config.py               
    ├── engine.py                
    ├── utils.py                 
    ├── metrics.py               
    ├── augment.py               
    ├── data/
    │   ├── transforms.py        
    │   ├── datasets.py          
    │   └── superpixel.py        SLIC and label-guided refinement
    ├── losses/
    │   ├── dice.py              
    │   └── domain.py           
    └── models/
        ├── resnet.py            
        ├── encoder.py           
        ├── modules.py           
        ├── extractor.py         
        └── propagation.py       
```

## Data

```
dataset/
├── WHDLD/
│   ├── images/*.jpg
│   ├── labels/*.png
│   └── image_list.txt
└── GID5_new/
    ├── images/*.tif
    ├── labels/*.tif
    ├── shot_GID5_new/5_shot{k}.csv
    └── query_GID5_new/5_query{k}.csv
pretrained_model/resnet50-19c8e357.pth
```

Generate the shot / query lists:

```bash
python tools/split_shot_query.py --root dataset/GID5_new --shot 5 --num-classes 5 --num-splits 100
```

## Usage

### 1. Pre-training

```bash
python pretrain.py
python pretrain.py --pretrain-epochs 20 --pretrain-lr 1e-4
python pretrain.py --source-val-list val_list.txt
```

### 2. Adaptation

```bash
python adapt.py
python adapt.py --shot 5 --num-splits 5 --epochs 10
python adapt.py --init-from snapshots/SADL/pretrain_epoch12.pth
python adapt.py --init-from "" --gpu 1 --save-pred true
```