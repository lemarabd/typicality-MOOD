# Typicality excels Likelihood for Unsupervised Out-of-Distribution Detection in Medical Imaging

This repository represents the code of the paper
_[Typicality Excels Likelihood for Unsupervised
Out-of-Distribution Detection in Medical
Imaging](https://openreview.net/pdf?id=a5Z1p2n7CN)_.


The code is adapted from [here](https://github.com/A-Vzer/WaveletFlowPytorch).

## Data

The datasets used in this work can be accessed below:

| Dataset name | Imaging modality | Download link |
|----------|----------|----------|
|   ISIC  |  Dermoscopic RGB images  |   [Link](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)  |
|   COVID-19  |   Chest X-Ray  |   [Link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)  |
|   Pneumonia  |   Chest X-Ray  |   [Link](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)  |
|   HeadCT |   Brain CT (2D)  |   [Link](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)  |

## Dependencies

Setup the conda environment with `environment.yml`:
```sh
conda env create -f environment.yml
```
Active the `typicalmood` conda environment:
```sh
source activate
conda activate typicalmood
```

## Training

Train the model on the ISIC dataset:

```sh
python train.py --data isic --batch_size 8 --alpha 2.0 --patience 10 
```

Other arguments can be seen by calling: 
```python train.py -h```.

The configuration of the GLOW model can be adjusted in `config_glow.py`. The hyperparameters used to train on ISIC can be adjusted in `isic.py`.

