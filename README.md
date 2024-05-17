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

pip install torchmetrics==0.9
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
Note: After cloning from original repo, need to modify: Device, Config of graphgym

### Additional modification
```python
File: /miniconda3/envs/graph-mamba/lib/python3.9/site-packages/torch_geometric/graphgym/config.py 

Modification: CN() -> CN(new_allowed=True)
```

```python
File: /miniconda3/envs/graph-mamba/lib/python3.9/site-packages/torch_geometric/graphgym/utils/device.py

Modification: 
def auto_select_device():
    r"""Auto select device for the current experiment."""
    if cfg.accelerator == 'auto':
        if torch.cuda.is_available():
            try:
                cfg.devices = 1
                # auto select device based on memory usage
                gpu_memory = get_gpu_memory_map()
                cfg.accelerator = f'cuda:{np.argmin(gpu_memory)}'
                cfg.device = f'cuda:{np.argmin(gpu_memory)}'
            except:
                cfg.accelerator = 'cuda'
                cfg.device = 'cuda'
        else:
            cfg.accelerator = 'cpu'
            cfg.devices = None
            cfg.device = 'cpu'
```

## Pre-Experiment
### Usage of Wandb
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

## Experiment
### Running command
```python
python main.py --cfg configs/Mamba/Learn_rank/brain_mamba.yaml
```
### Usage of config.yaml
```
wandb -> wandb setting

dataset:
  format: Brain -> dataset
  name: brain -> dataset

train:
  enable_ckpt -> whether save ckpt 
  auto_resume -> whether use ckpt
  # ckpt_period is always 1 greater than epoch_resume

model:
  captum -> whether use captum to explain
  gnn_explainer -> whether use gnn_explainer to explain
  mamba_interpret -> whether use the mamba inherent parameter to explain
```

### Repo structure
```
datasets
- brain : store the brain dataset
graphgps
- config : the config files
- loader
  - master_loader.py : the interface for loading datasets, when incorporating a new dataset, need to add a new function and a if branch.
  - dataset : store all dataset class
- network
  - gps_model.py : the mamba model
- layer
  - gps_layer.py : the mamba layer, the code for saving mamba parameters is inside
```

### TODO
- Add more benchmarks
  - require modifying the loader dir in graphgps: in master_loader.py and dataset
  - at least 3
- Add more baselines
  - current we have gatedGCN, transformer, Mamba
  - at least 5
- Run MoE-mamba on brain dataset 
  - implement the newly implemented moe-mamba
  - design a method to retrieve explanable parameters for all experts
- Explain experiement design and visualization