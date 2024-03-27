# Introduction
This repo is forked from Graph-mamba paper, for reproduction use.

# Guideline
## Installation
```
conda create -n graph-mamba python==3.9
conda activate graph-mamba
pip install torch==1.13.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_geometric==2.0.4

pip install transformers==4.36.2
pip install torchmetrics==0.10.3
pip install openbabel-wheel
pip install fsspec 
pip install rdkit
pip install pytorch-lightning yacs
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

pip install causal-conv1d==1.0.2
pip install mamba-ssm==1.0.1
```

## Experiment
### Step 1: Modifying configs

### Step 2: Running bash

### Step 3: Share Wandb
#### How to register for a wandb account
#### How to connect to wandb
#### How to share running logs