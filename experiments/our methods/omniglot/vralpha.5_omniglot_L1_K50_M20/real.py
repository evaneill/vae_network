from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, Tensor as T
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.distributions.multinomial import Multinomial
import datetime
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import logging
import math

batch_size = 20
test_batch_size = 32
epochs = 501
seed = 1
log_interval = 100
log_test_value = 100
K = 50
learning_rate = 5e-4
discrete_data = True
num_rounds = 6
cuda = torch.cuda.is_available()

alpha = .5
model_type = 'vralpha'

torch.manual_seed(seed)

data_name = 'omniglot'

device = torch.device("cuda" if cuda else "cpu")

if model_type!="general_alpha":
	model_name=model_type
else:
	model_name = model_type+str(alpha)

logging_filename = f'{model_name}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)

# Load data with random initialized train/test split
if os.environ.get('CLOUDSDK_CONFIG') is not None:
    fpath = "/content/drive/My Drive/data/chardata.mat"
else:
    fpath = os.path.abspath('../../../data/chardata.mat')

data = loadmat(fpath)

# From iwae repository
data_train = data['data'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
data_test = data['testdata'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')

data_train_t, data_test_t = T(data_train), T(data_test)

# Define likelihood functions
def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / (torch.exp(logstd)+torch.tensor(1e-19))) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi))

def compute_log_probabitility_bernoulli(theta, obs, axis=1):
    return torch.sum(obs*torch.log(theta) + (1-obs)*torch.log(1-theta), axis)


# Define the model
class omniglot1_model(nn.Module):
    def __init__(self):
        super(omniglot1_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = K

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, test=False, alpha=alpha):
        # data = (N,560)
        if model_type == 'vae':
            alpha = 1
        elif model_type in ('iwae', 'vrmax'):
            alpha = 0
        else:
            # use whatever alpha is defined in hyperparameters
            if abs(alpha - 1) <= 1e-3:
                alpha = 1

        data_k_vec = data.repeat_interleave(K, 0)

        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = torch.sum(-0.5 * z ** 2, 1) - .5 * z.shape[1] * T.log(torch.tensor(2 * np.pi))
        decoded = model.decode(z)  # decoded = (pmu, plog_sigma)
        log_p = compute_log_probabitility_bernoulli(decoded, data_k_vec)
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or test == True:
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
        elif model_type == 'vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1) * 1 / K
        elif model_type == 'general_alpha' or model_type == 'vralpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)
        elif model_type == 'vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1, keepdim=True).values

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        if model_type == 'vralpha' and not test:
            sample_dist = Multinomial(1, ws_norm)
            ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
        else:
            ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)

        if model_type in ["general_alpha", "vralpha"] and not test:
            ws_sum_per_datapoint /= (1 - alpha)

        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, logstd, loss


# train and test functions
def train(round_num,epoch,optimizer):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        recon_batch, _, _, loss = model.compute_loss_for_batch(data, model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logging.info('Round {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                round_num,epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Round {}, Epoch: {} Average loss: {:.4f}'.format(
          round_num,epoch,  train_loss / len(train_loader.dataset)))
    logging.info('====> Round {}, Epoch: {} Average loss: {:.4f}'.format(
          round_num,epoch,  train_loss / len(train_loader.dataset)))

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(round_num,epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            _, _, _, loss = model.compute_loss_for_batch(data, model, K=5000,test=True)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,28),
                                      recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(round_num)+'_'+str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Round {} Test set loss: {:.4f}'.format(round_num,test_loss))
    logging.info('====> Round {} Test set loss: {:.4f}'.format(round_num,test_loss))

# Initialize a model and data loaders
model = omniglot1_model().to(device)

train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True)

# Call the training shenanigans
if torch.cuda.is_available():
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs('results', exist_ok=True)
model = omniglot1_model().to(device)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for i in range(num_rounds):
    current_round_lr = learning_rate*math.pow(10,-i/7)
    optimizer = optim.Adam(model.parameters(), lr=current_round_lr)
    print(f"========Current round LR: {current_round_lr}=======")
    logging.info(f"========Current round LR: {current_round_lr}=======")
    print(f"======== About to train for {3**i} epochs =========")
    logging.info(f"======== About to train for {3**i} epochs =========")
    for epoch in range(1,3**i + 1):
        train(i,epoch,optimizer)
        if epoch == 1 or epoch % 3**(i-1) ==0:
            _test(i,epoch)
            with torch.no_grad():
                z2 = torch.randn(64, 50).to(device)
                sample = model.decode(z2).cpu()
                save_image(sample.view(64, 1, 28, 28),
                            'results/sample_' +str(i)+'_'+ str(epoch) + '.png')

_test(i,epoch)
print(datetime.datetime.now())
logging.info(datetime.datetime.now())
print("Training finished")
logging.info("Training finished")