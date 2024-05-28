# GNN-Asymptotically-Constant

This repository contains the official code base of the paper **[Almost Surely Asymptotically Constant Graph Neural Networks](https://arxiv.org/abs/2403.03880)**.

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.0.0, Cuda 11.8, PyG version 2.3.0, and torchmetrics.

```bash
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.3.0
pip install ogb
```

## Datasets

### datasets

ER, LogER and InverseER represent the $`ER(n, p(n) = 0.1)`$, $`ER(n, p(n) = \frac{\log{n}}{n})`$
and $`ER(n, p(n) = \frac{1}{50n})`$ distributions.

## Running

The script we use to run the experiments is ``./main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

The parameters of the script are:

- ``--dataset``: name of the dataset.
The available options are: ER, LogER, InverseER, SBM, BA, Tiger1k, Tiger5k, Tiger10k, Tiger25k and Tiger90k.

- ``--graph_size``: the graph size.
- ``--num_graph_samples``: the number of different graph size samples taken. 
- ``--rw_pos_length``: the maximal length of the random walk in the Random Walk Positional Encoding.
- ``--model_type``: the type of model that is used.
The available options are: MEAN_GNN, GCN, GAT and GPS.

- ``--num_layers``: the network's number of layers.
- ``--in_dim``: the network's input dimension.
- ``--hidden_dim``: the network's hidden dimension.
- ``--output_dim``: the network's output dimension.
- ``--pool``: name of the graph pooling.

- ``--seed``: a seed to set random processes.
- ``--gpu``: the number of the gpu that is used to run the code on.
  
## Example running

To perform experiments over the LogER dataset with a MEAN_GNN with 3 layers, output dimension of 5 and an input and hidden dimension of 128.  See an example for the use of the following command: 
```bash
python -u main.py --dataset LogER --model_type MEAN_GNN --in_dim 128 --hidden_dim 128 --out_dim 5 --num_layers 3 --seed 0
```
