# About
This repository contains the code we used to replicate the paper [[1]](https://arxiv.org/pdf/1602.02311.pdf) on using Renyi alpha-divergences for Variational Inference.
Our implementation is based on PyTorch and we extended the [basic example of a VAE](https://github.com/pytorch/examples/blob/master/vae/main.py) from the PyTorch library.

A commented version of our code that works on the MNIST and FashionMNIST dataset can be found in the file `example_models.py`. The rest of our code (which we ran on and optimized for) Google Colab is in the `experiments` directory.
## Running an experiment
###Without any set up

Run the `Run_experiments_on_MNIST_FashionMNIST.ipynb` notebook in Google Colab.
The code in this notebook works on the MNIST and FashionMNIST dataset.

### Else: Get the data
You require the data, which can either be put into `data/` in the repository or into your google drive under a `data/` folder in the root. The default behavior in the code used in Colab is to assume that the data lies on your google drive, and would have to be altered at runtime.

In either case, it should be organized simply as follows:

`data/` directory (either in repo or google drive) must contain:
  - `freyfaces.pkl`  - FreyFace data (from the repo [here](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl))
  - `chardata.mat` - OMNIGLOT data
  - `caltech101_silhouettes_28.mat` - Silhouettes data

`MNIST` data is downloaded on the first call - it isn't stored here.

### Set the environment, hyperparameters, and runtime

Each directory under `experiments/exact_replication/<dataset_name>` and `experiments/replication_increased_lr/<dataset_name>` holds a collection of `.py` files that holds code that will have to be ran in a Colab cell. When I ran the experiments, I executed the code blocks in the following order:

1. `imports.py` 
2. `train_and_hyperparameters.py`  
3. `data_import.py`
4. `model.py`
5. `likelihoods.py`
6. `train_and_test.py`
7. `runtime.py`

The order of these cells is largely independent, except that `imports` should come first and `runtime` should come last. 

For convenience, we provided a notebook `Rerun_experiments.ipynb` that clones the Github repository and automatically imports all the needed scripts to run an experiment. It only requires the specification of which experiment to run.

### Log the experiment

Every experiment sets up a logger that will store the training logs and any training schedule messages, as well as intermittent image sampling and reconstructions. If the experiment produces a `results/` folder in the Colab file directory with reconstruction and sample images, it will be necessary to run:
```
fstring = f'{model_name}_{data_name}_K{K}_M{batch_size}'
!rm -r {fstring}

!mkdir {fstring}
!mkdir {fstring}/samples
!mkdir {fstring}/recons
!mv results/reconstruction_*  {fstring}/recons/
!mv results/sample_* {fstring}/samples/
!rm -r results
```
If `data_name` isn't already defined, it's either `silhouettes`, `mnist`, `omniglot`, `fashion` or `freyfaces`. 

`model_name` is always `model_type` except for some experiments (`vralpha` and `general_alpha` created late in the project) in which `model_name = model_type+str(alpha)`.

Lastly you can run this code block which should zip up everything onto your drive, including a copy of the final model:
```

from zipfile import ZipFile
import os

with open(f'{model_name}_{data_name}_K{K}_M{batch_size}/{model_name}_{data_name}_K{K}_M{batch_size}.pt','wb') as f:
  torch.save(model,f)

fstring = f'{model_name}_{data_name}_K{K}_M{batch_size}'
!mv {fstring}.log {fstring}/
with ZipFile(f'drive/My Drive/experiment results/{model_name}_{data_name}_K{K}_M{batch_size}.zip','w') as f:
  f.write(f'{model_name}_{data_name}_K{K}_M{batch_size}/{model_name}_{data_name}_K{K}_M{batch_size}.pt')
  f.write(f'{model_name}_{data_name}_K{K}_M{batch_size}/{model_name}_{data_name}_K{K}_M{batch_size}.log')
  for img in os.listdir(f'{model_name}_{data_name}_K{K}_M{batch_size}/samples'):
    if img.endswith('.png'):
      f.write(f'{model_name}_{data_name}_K{K}_M{batch_size}/samples/'+img)

  for img in os.listdir(f'{model_name}_{data_name}_K{K}_M{batch_size}/recons'):
    if img.endswith('.png'):
      f.write(f'{model_name}_{data_name}_K{K}_M{batch_size}/recons/'+img)
```

This is admittedly not the most elegant way to do things, but it provides a faithful means of recreation. Our initial efforts to have a more general purpose package to encapsulate all of this failed. This falls into the category of 'what we would have improved, if we had more time'.

    
