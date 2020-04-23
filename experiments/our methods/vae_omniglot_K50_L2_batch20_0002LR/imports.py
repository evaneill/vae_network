from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, Tensor as T
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import logging
import math

# import torch_xla
# import torch_xla.core.xla_model as xm