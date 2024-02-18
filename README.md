
 ST-MTM (Seasonality-Trend Masked Time-series Model)
===============

### This repository provides PyTorch implementation of ST-MTM from the paper "ST-MTM: Masked Time Series Modeling with Seasonal-Trend Decomposition for Time Series Forecasting"

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




