# CUDA-accelerated t-SNE using PyTorch
PyTorch implementation of the t-stochastic neighbor embedding algorithm described in [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf). 

While CUDA support is not required for this library, the best performance can be achieved when this library is used on a system with CUDA support.

## Installation

Requires Python 3.7

### Install via Pip

```bash
pip3 install tsne-torch
```

### Install from Source

```bash
git clone https://github.com/palle-k/tsne-pytorch.git
cd tsne-pytorch
python3 setup.py install
```

## Usage

```python
from tsne_torch import TorchTSNE as TSNE

X = ...  # shape (n_samples, d)
X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(X)  # returns shape (n_samples, 2)
```

## Command-Line Usage

```bash
python3 -m tsne_torch --xfile <path> --yfile <path>
```

## Example

This is our result compared to the result of the author's Python implementation on a subset of the MNIST dataset:

* PyTorch result

![pytorch result](https://github.com/palle-k/tsne-pytorch/raw/master/images/pytorch.png)
* python result

![python result](https://github.com/palle-k/tsne-pytorch/raw/master/images/python.png)

## Credit
This code highly inspired by 
* author's python implementation code [here](https://lvdmaaten.github.io/tsne/).
