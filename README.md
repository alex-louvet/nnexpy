# NNexpy (A bunch of python script and classes to experiment with homology, knot theory and neural networks) 

## Introduction

This repository is a compilation of my work during my master course at Kyushu Institute of Technology. A clone of the repository is also available on [KIT Honda Lab's github](https://github.com/honda-lab-kit/nn-expressiveness). The purpose of this repository is to provide with some python code to get started with homology and knot theory as a mean to evaluate neural networks' expressivity.

## Requirement

To run this code you need python3 as well as some libraries. They can be installed unsing Conda and I personnaly used a conda environment which [environment file](https://github.com/Spiilgriim/nn-expressiveness/blob/master/environment.yml) is available in this repository.

To setup a functional conda environment with conda cli you simply need to use the following command:

```shell
conda env create -f environment.yml
conda activate nexpy
```

For more information refer to [conda documentation](https://docs.conda.io/projects/conda/en/latest/index.html) and in particular the section about [managing environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

## Usage

To use this code, copy the nnexpy folder in your python project and import `nnexpy`

```python
import nnexpy
```

NNexpy contains the following classes:

* [Bounds](https://github.com/Spiilgriim/nnexpy/blob/master/docs/Bounds.md)
* [DataDescriptor](https://github.com/Spiilgriim/nnexpy/blob/master/docs/DataDescriptor.md)
* [DataInstance](https://github.com/Spiilgriim/nnexpy/blob/master/docs/DataInstance.md)
* [DataPoint](https://github.com/Spiilgriim/nnexpy/blob/master/docs/DataPoint.md)
* [KnotDescriptor](https://github.com/Spiilgriim/nnexpy/blob/master/docs/KnotDescriptor.md)
* [NetworkGenerator](https://github.com/Spiilgriim/nnexpy/blob/master/docs/NetworkGenerator.md)
* [RandomNetwork](https://github.com/Spiilgriim/nnexpy/blob/master/docs/RandomNetwork.md)

## Examples

The [`examples`](https://github.com/Spiilgriim/nnexpy/tree/master/examples) folder is a collection of the various scripts I have been creating during my research. Here follows a short description of all of them.

* [`ai_playground.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/ai_playground.py): Generates a few `DataInstance` with different homology and then call `rasScript` to train batch of networks on these different datasets.
* [`analyse_network.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/analyse_network.py): Computes betti numbers and makes the PCA to obtain a trajectory from Keras + Tensorflow neural network.
* [`bettiScript.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/bettiScript.py): Computes Betti numbers of Keras + Tensorflow neural networks on a `DataDescriptor`
* [`oldBottleneckScript.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/bottleneckScript.py): Computes Betti numbers of Keras + Tensorflow neural networks on a `DataDescriptor`
* [`compare_results.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/compare_results.py): Script to use `bettiScript`, `oldBettiScript` and `oldBottleneckScript` as subprocesses
* [`oldBettiScript.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/oldBettiScript.py): Another version of betti number computation of Keras + Tensorflow neural networks on a `DataDescriptor`
* [`rasScript.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/rasScript.py): Trains a Keras + Tensorflow neural network on a given `DataInstance`
* [`test_random_networks.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/test_random_networks.py): Generate PCA of random networks with an imput trajectory
* [`visualize_results.py`](https://github.com/Spiilgriim/nnexpy/blob/master/examples/visualize_results.py): Plot the predicted data of Keras + Tensorflow networks on a `DataInstance`
