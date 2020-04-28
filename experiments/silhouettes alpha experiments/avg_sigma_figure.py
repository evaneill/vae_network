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


import matplotlib
from matplotlib import pyplot as plt

seed = 1
K = 50 
torch.manual_seed(seed)
alpha=None # Doesn't matter here

# Define the model
class silhouettes_model(nn.Module):
    def __init__(self):
        super(silhouettes_model, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, 200)
        self.fc22 = nn.Linear(500, 200) 

        self.fc3 = nn.Linear(200, 500)
        self.fc4 = nn.Linear(500, 784) 

        self.K = K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logstd,test=False):
        std = torch.exp(logstd)
        if test==True:
          eps = torch.zeros_like(mu)
        else:
          eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z,test=False):
        h3 = torch.tanh(self.fc3(z))

        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K,test=False,alpha=alpha):
        # data = (N,560)
        if model_type=='vae':
            alpha=1
        elif model_type in ('iwae','vrmax'):
            alpha=0
        elif model_type=='general_alpha' or model_type=='vralpha':
            # use whatever alpha is defined in hyperparameters
            if abs(alpha-1)<=1e-3:
                print("Can't plug 1 into general alpha formula, use model_type='vae' instead")
                raise Exception

        data_k_vec = data.repeat_interleave(K,0)

        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = torch.sum(-0.5 * z ** 2, 1)-.5*z.shape[1]*T.log(torch.tensor(2*np.pi)) 
        decoded = model.decode(z) # decoded = (pmu, plog_sigma)
        log_p = compute_log_probabitility_bernoulli(decoded,data_k_vec)
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or test==True:
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1)*1/K
        elif model_type=='general_alpha' or model_type=='vralpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True).values

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        
        if model_type=='vralpha' and not test:
            sample_dist = Multinomial(1,ws_norm)
            ws_sum_per_datapoint = log_w_matrix.gather(1,sample_dist.sample().argmax(1,keepdim=True))
        else:
            ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        
        if model_type in ["general_alpha","vralpha"] and not test:
            ws_sum_per_datapoint/=(1-alpha)
        
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, loss

def make_plot():

	### Load the models

	modelneg500 = torch.load('vralpha-500_silhouettes_K50_M20/vralpha-500_silhouettes_K50_M20.pt',map_location = torch.device('cpu'))
	modelplus500 = torch.load('vralpha500_silhouettes_K50_M20/vralpha500_silhouettes_K50_M20.pt',map_location = torch.device('cpu'))

	### Load the data and minibatch
	fpath = os.path.abspath('../../data/caltech101_silhouettes_28.mat')

	data = loadmat(fpath) 
	data = 1-data.get('X')

	np.random.seed(seed)
	np.random.shuffle(data)

	num_train = int(.9* data.shape[0])

	data_train = data[:num_train]
	data_test = data[num_train:]

	data_train_t, data_test_t = T(data_train), T(data_test)

	# Use a large minibatch size of 1000
	train_loader = DataLoader(TensorDataset(data_train_t),batch_size=1000,shuffle=True,pin_memory=True)

	minibatch = next(iter(train_loader))[0] # Shuffle then give 1000 samples


	decodedneg500, muneg500, logstdneg500 = modelneg500.forward(minibatch)
	decodedplus500, muplus500, logstdplus500 = modelplus500.forward(minibatch)

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

	# sorted_avgmu_list_plus500 = muplus500.mean(axis=1).index_select(0,orderplus500.indices).tolist()
	# sorted_avgmu_list_neg500 = muneg500.mean(axis=1).index_select(0,orderneg500.indices).tolist()  

	matplotlib.rc('font',size=16)
	plt.style.use('seaborn-darkgrid') 

	plt.plot(sorted_avgsigma_list_neg500,label='alpha=-500')
	plt.plot(sorted_avgsigma_list_plus500,label='alpha=500')

	# plt.plot(sorted_avgmu_list_neg500,'g--')
	# plt.plot(sorted_avgmu_list_plus500,'b--')

	plt.xlabel('Index of latent dimension z, ordered')
	plt.ylabel('Average Standard Deviation')

	plt.subplots_adjust(top=.95,bottom=.15,left=.10,right=.95) 

	plt.legend()

	plt.show()

if __name__=='__main__':
	make_plot()        
