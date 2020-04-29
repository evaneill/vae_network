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
from timeit import default_timer as timer
batch_size = 128
epochs = 141
seed = 1
log_interval = 1000
testing_frequency = 20
K = 50
learning_rate = 1e-3
discrete_data = True
alpha = 0
cuda = torch.cuda.is_available()
test_batch_size = 32
model_type = 'vrmax'
torch.manual_seed(seed)

data_name = 'mnist'

logging_filename = f'{model_type}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)


def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi))

def compute_log_probabitility_bernoulli(obs, p, axis=1):
    return torch.sum(p*torch.log(obs) + (1-p)*torch.log(1-obs), axis)

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
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).min(axis=1, keepdim=True).values
            return 0, 0, 0, -torch.sum(log_w_matrix)
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, logstd, loss

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        recon_batch, _, _, loss = model.compute_loss_for_batch(data, model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch,  train_loss / len(train_loader.dataset)))
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch,  train_loss / len(train_loader.dataset)))
    train_losses.append(train_loss / len(train_loader.dataset))


# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            _, _, _, loss = model.compute_loss_for_batch(data, model, 5000, testing_mode=True)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Test set loss: {:.4f}'.format(test_loss))
    logging.info('====> Test set loss: {:.4f}'.format(test_loss))
    test_losses.append(test_loss)

train_losses = []
test_losses = []

device = torch.device("cuda" if cuda else "cpu")

model = mnist1_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=test_batch_size, shuffle=True)

if torch.cuda.is_available():
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs('results', exist_ok=True)
print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for epoch in range(1, epochs + 1):
    start = timer()
    train(epoch)
    end = timer()
    print(f"time taken for 1 epoch: {end-start} s")
    if epoch % testing_frequency == 1:
        _test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 50).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
print("Training finished")
logging.info("training finished")