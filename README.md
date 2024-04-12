
 ST-MTM (Seasonality-Trend Masked Time-series Model)
===============

### This repository provides PyTorch implementation of ST-MTM from the paper "ST-MTM: Masked Time Series Modeling with Seasonal-Trend Decomposition for Time Series Forecasting"

# Tables for response to reviewers
We provide the experimental results requested by reviewers. The extended experiments and analyses will be updated in the revised manuscript. We look forward to any further comments and questions.

### 1. Comparative evaluation with state-of-the-art MTM methods on Exchange and PEMS08 datasets
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/2cc8a5ed-3e26-47a0-b02c-6fa67192b641>
</p>

### 2. Running time analysis at training phases on ETTh1 (seconds)
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/c9fa1f65-8bdd-4261-93ac-7e0a114dc3fd>
</p>

### 3. Frequently used notations
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/13859073-0c29-4bfe-a3e3-54e3a23712a0>
</p>

### 4. Comparative evaluation with SCNN on ETTh1 and ETTh2 datasets
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/d5d6e3a0-1250-4595-b104-c0dbfdfad39d>
</p>

### 5. Sensitivity analysis on sub-series length on ETTh1 and ETTh2 datasets
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/706d5242-da16-4048-9522-ec7a3d354c59>
</p>

### 6. Sensitivity analysis on batch size on ETTh1 and ETTh2 datasets
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/42451519-5b52-4eee-aa2f-0e3e14aaf447>
</p>

### 7. Sensitivity analysis on temperature on ETTh1 and ETTh2 datasets
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/9102f8d3-1dfc-4179-b921-fa2218bff079>
</p>

### 8. Comparative evaluation with state-of-the-art MTM methods and naive forecasts on Exchange dataset
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/9a6926bf-ba13-4d4f-8e7a-5ad176bd33b2, width = "70%" height = "70%">
</p>

### 9. Comparative evaluation with PatchTST at L=512 on ETTh2 dataset
<p align="center">
  <img src=https://github.com/crosslongfitstone/ST-MTM/assets/159516581/01388f10-c04f-4961-ac94-489620d7a69f, width = "60%" height = "60%"> 
</p>



# Requirements

- Python 3.9.0
- torch == 2.0.1
- numpy==1.24.3
- pandas==1.5.3
- scikit-learn==1.2.2
- matplotlib==3.7.1
- tensorboardX==2.6.2.2

Dependencies can be installed using the following command:

    pip install -r requirements.txt

# Getting Started

## 1. Prepare Data

All benchmark datasets can be obtained from [Google Drive](https://drive.google.com/drive/folders/1NL8AeO5-C9NFZkGT-FlPMeORYJYOv2x-?usp=sharing), and arrange the folder as:

    ST-MTM/
    |-- datasets/
        |-- ETTh1.csv
        |-- ETTh2.csv
        |-- ETTm1.csv
        |-- ETTm2.csv
        |-- Weather.csv
        |-- Electricity.csv
        |-- SP500.csv

## 2. Experimental reproduction

- We provide the scripts for pre-training and finetuning for each dataset with the best hyper-parameters in our experiment at `./scripts/`.

### 2-1. Pre-training

Pre-training ST-MTM for each dataset can be implemented through the provided scripts in `./scripts/pretrain/`. For example, to pre-train ST-MTM for ETTh1 dataset:

    bash scripts/pretrain/ETT_script/ETTh1.sh

### 2-2. Fine-tuning

After pre-training ST-MTM for the dataset, fine-tuning ST-MTM for forecasting across various lengths can be implemented through the provided scripts in `./scripts/finetune/`. For example, to fine-tune ST-MTM for ETTh2 dataset:

    bash scripts/finetune/ETT_script/ETTh1.sh

### 2-3. Pre-training and fine-tuning at once

To implement pre-training and fine-tuning sequentially, the scripts in `./scripts/`. For example, to perform both steps at once for electricity dataset:

    bash scripts/run_electricity.sh


