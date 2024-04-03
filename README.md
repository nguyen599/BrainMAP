# Introduction
This repo is forked from Graph-mamba paper, for reproduction use.

# Guideline
## Installation
```
conda create -n graph-mamba python==3.9
conda activate graph-mamba
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric[graphgym]

pip install torchmetrics==0.9 (important)
pip install openbabel-wheel
pip install fsspec 
pip install rdkit
pip install pytorch-lightning yacs
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
pip install deepspeed
pip install torch_ppr

pip install causal-conv1d==1.0.2
pip install mamba-ssm==1.0.1
```
Note: After cloning from original repo, need to modify: Device, Config

## Experiment
### Step 1: Share Wandb
#### 1. How to start a wandb account?
Sign up an account for this website.
```
https://wandb.ai/site
```

On the `https://wandb.ai/home` page, around the right top corner thereis a blue (green) button with text `Invite Teammates` on it.

Click on that button, then create a new account with any name,  **work** and **research** are both OK (**research** are free to use). 

Then create two new projects called `GNN-Benchmark-RandomSeed` and `LRGB-Benchmark-RandomSeed`

#### 2. How to connect to wandb
Open your remote, under the repo of `Graph Mamba`.

In command line, type in
```
pip install wandb
wandb login
```
Then it asks you to give the API key, where you could type in the key from `https://wandb.ai/{your-team-name}`.

Then, modify the `configs\Mamba\{running-config-file}.yaml` in local repo.
```
wandb:
  use: True
  project: LRGB-Benchmark-RandomSeed
  entity: vjd5zr -> {your-username} (Click on your profile on the right top corner, then the second name under your avatar)
```

#### 3. How to share running logs
Add team members to the team account. On the left column in `https://wandb.ai/{your-team-name}`, click on `Invite Team Members`. (Here please type in my account vjd5zr@virginia.edu)

### Step 2: Running experiments
Running all of the commands in `run.sh`.