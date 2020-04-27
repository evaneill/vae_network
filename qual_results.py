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
import numpy as np
import logging

K = 50
discrete_data = True
model_type = 'no'
def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi))

def compute_log_probabitility_bernoulli(theta, obs, axis=1):
    return torch.sum(obs*torch.log(theta+1e-18) + (1-obs)*torch.log(1-theta+1e-18), axis)

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

    def compute_loss_for_batch(self, data, model, K=K, testing_mode=False):
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
        elif model_type == 'vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1, keepdim=True).values
            return 0, 0, 0, -torch.sum(log_w_matrix)
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, logstd, loss

device = 'cuda'

model = mnist1_model()
model.load_state_dict(torch.load('vrmax_fashion_L1_K50_M128.pt'))  # Choose whatever GPU device number you want
model.to(device)

model2 = mnist1_model()
model2.load_state_dict(torch.load('iwae_fashion_L1_K50_M128.pt'))  # Choose whatever GPU device number you want
model2.to(device)

model3 = mnist1_model()
model3.load_state_dict(torch.load('vae_fashion_L1_K50_M128.pt'))  # Choose whatever GPU device number you want
model3.to(device)
with torch.no_grad():
    sample = torch.randn(64, 50).to(device)
    result1 = model.decode(sample).cpu()
    save_image(result1.view(64, 1, 28, 28),
               f'results/sample_vrmax.png')
    result2 = model2.decode(sample).cpu()
    save_image(result2.view(64, 1, 28, 28),
               f'results/sample_iwae.png')
    result3 = model3.decode(sample).cpu()
    save_image(result3.view(64, 1, 28, 28),
               f'results/sample_vae.png')

    test_batch_size = 32
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=test_batch_size, shuffle=True)

    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   'results/reconstruction_vrmax.png', nrow=n)

        recon_batch2, mu, logvar = model2(data)
        comparison = torch.cat([data[:n],
                                recon_batch2.view(test_batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   'results/reconstruction_iwae.png', nrow=n)
        recon_batch3, mu, logvar = model3(data)
        comparison = torch.cat([data[:n],
                                recon_batch3.view(test_batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   'results/reconstruction_vae.png', nrow=n)
        break