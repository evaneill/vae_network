# vae_network
Try to recreate VAE implementation from project paper using pytorch

[Paper in reference](https://arxiv.org/pdf/1602.02311.pdf) - Renyi divergence.

### utils.py

Contains a `Loader` class that can be instantiated with the name of a dataset and load data into training and test sets in np.ndarray form.

E.g. 
```
from utils import Loader

my_loader = Loader('MNIST')

train, test = my_loader.load()
```
In order to work, you gotta have the data to load in the first place:

`data/` directory must contain:
  - `freyfaces.pkl`  - FreyFace data
  - `chardata.mat` - OMNIGLOT data
  - `caltech101_silhouettes_28.mat` - Silhouettes data
  - `MNIST/` - MNIST data broken up in different files. Labels are irrelevant but will break my fragile functions if not present
    - `t10k-images-idx3-ubyte`
    - `t10k-labels-idx1-ubyte`
    - `train-images-idx3-ubyte`
    - `train-labels-idx1-ubyte`
   - `BinaryMNIST/` - Binarized MNIST data from iwae repository (https://github.com/yburda/iwae) sans readme
      - `binarized_mnist_test.amat`
      - `binarized_mnist_train.amat`
      - `binarized_mnist_valid.amat`
    
