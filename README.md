# Learning Low-Rank Feature for Thorax Disease Classification


## Installing Requirements

Our codebase follows the [MAE Official](https://github.com/facebookresearch/mae) and uses some additional packages.
You may use **one of** the following commands to build environments with `Conda` and `Pip`.

Conda:
```
conda create -n medical_mae -f medical_mae.yml 
```

Pip:
```
conda create -n medical_mae python=3.8
conda activate medical_mae
pip install -r requirements.txt 
```


## Preparing Datasets:

The MIMIC-CXR, CheXpert, and ChestX-ray14 datasets are public available on their official sites. You can download or request the access to them under the agreements.

You may also download them through the following links for research only and follow the official agreements.

MIMIC-CXR (JPG): https://physionet.org/content/mimic-cxr-jpg/2.0.0/

CheXpert (v1.0-small): https://www.kaggle.com/datasets/ashery/chexpert

ChestX-ray14 : https://www.kaggle.com/datasets/nih-chest-xrays/data



## Pre-training on ImageNet or Chest X-rays

The pre-training instruction is in [PRETRAIN.md](https://github.com/lambert-x/medical_mae/blob/main/PRETRAIN.md).




