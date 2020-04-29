from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, Tensor as T
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable, detect_anomaly
from torch.distributions.multinomial import Multinomial
import datetime
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import logging
import math

from matplotlib import pyplot as plt
import matplotlib

seed = 1
K = 50 
torch.manual_seed(seed)
alpha=None # Doesn't matter here

class mnist1_model(nn.Module):
    def __init__(self):
        super(mnist1_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = K
    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, testing_mode=False,alpha=alpha):
        # data = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False), torch.zeros_like(z, requires_grad=False))
        decoded = model.decode(z)
        if discrete_data:
            log_p = compute_log_probabitility_bernoulli(decoded, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p = compute_log_probabitility_gaussian(decoded, data_k_vec, torch.zeros_like(decoded))
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or testing_mode:
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1)*1/K
        elif model_type=='general_alpha' or model_type=='vralpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True).values

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        
        if model_type=='vralpha' and not testing_mode:
            sample_dist = Multinomial(1,ws_norm)
            ws_sum_per_datapoint = log_w_matrix.gather(1,sample_dist.sample().argmax(1,keepdim=True))
        else:
            ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        
        if model_type in ["general_alpha","vralpha"] and not testing_mode:
            ws_sum_per_datapoint/=(1-alpha)
        
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded,mu, logstd, loss

def make_plot(save_recons=True):

    ### Load the models

    # This is very poor form but yet here we are
    exec(open('vralpha-500_mnist_L1_K50_M128/model.py').read())

    # Pull in the saved models
    modelneg500 = torch.load('vralpha-500_mnist_L1_K50_M128/vralpha-500_mnist_K50_M128.pt',map_location = torch.device('cpu'))
    modelplus500 = torch.load('vralpha500_mnist_L1_K50_M128/vralpha500_mnist_K50_M128.pt',map_location = torch.device('cpu'))

    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1000, shuffle=True)

    minibatch = next(iter(train_loader))[0] # Shuffle then give 1000 samples

    decodedneg500, muneg500, logstdneg500 = modelneg500.forward(minibatch)
    decodedplus500, muplus500, logstdplus500 = modelplus500.forward(minibatch)

    if save_recons:
        i = 891 # This index can be adjusted to come up with new reconstruction samples to view
        comparisonneg500 = torch.cat([minibatch[i:i+8].view(-1,1,28,28),decodedneg500.view(1000, 1, 28, 28)[i:i+8]]) 
        comparisonplus500 = torch.cat([minibatch[i:i+8].view(-1,1,28,28),decodedplus500.view(1000, 1, 28, 28)[i:i+8]])

        comparison = torch.cat([minibatch[i:i+8].view(-1,1,28,28),decodedneg500.view(1000, 1, 28, 28)[i:i+8],decodedplus500.view(1000, 1, 28, 28)[i:i+8]])
        save_image(comparison.cpu(),'reconstruction.png', nrow=8)

        save_image(comparisonneg500.cpu(),'reconstructionneg500.png', nrow=8)
        save_image(comparisonplus500.cpu(),'reconstructionplus500.png', nrow=8)

    orderneg500 = logstdneg500.exp().mean(axis=0).sort(descending=True)
    orderplus500 = logstdplus500.exp().mean(axis=0).sort(descending=True)

    sorted_avgsigma_list_plus500 = orderneg500.values.tolist() 
    sorted_avgsigma_list_neg500 = orderplus500.values.tolist()

    sorted_avgmu_list_plus500 = muplus500.mean(axis=1).index_select(0,orderplus500.indices).tolist()
    sorted_avgmu_list_neg500 = muneg500.mean(axis=1).index_select(0,orderneg500.indices).tolist()  

    print(f"Std dev of means (alpha=-500): {np.mean(sorted_avgmu_list_neg500)}")
    print(f"Std dev of means (alpha=500): {np.mean(sorted_avgmu_list_plus500)}")

    matplotlib.rc('font',size=16)
    plt.style.use('seaborn-darkgrid') 

    plt.plot(sorted_avgsigma_list_neg500,label='alpha=-500') 
    plt.plot(sorted_avgsigma_list_plus500,label='alpha=500')


    # plt.plot(sorted_avgmu_list_neg500,'g--')
    # plt.plot(sorted_avgmu_list_plus500,'b--')

    plt.xlabel('Index of latent dimension z, ordered')
    plt.ylabel('Average Standard Deviation')

    # plt.subplots_adjust(top=.95,bottom=.15,left=.10,right=.95) 

    plt.legend()

    plt.show()

# Thanks stackoverflow https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Make pretty plots about latent distributions')
    parser.add_argument("--savimg", type=str2bool,const=True, nargs="?",default=True,help="save some image reconstructions")

    args = parser.parse_args()

    make_plot(save_recons = args.savimg)        
