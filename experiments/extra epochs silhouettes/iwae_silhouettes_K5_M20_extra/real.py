from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, Tensor as T
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable, detect_anomaly
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


batch_size = 20 #@param {type:"slider", min:1,max:200}
test_batch_size = batch_size
# testing_frequency=100 # Depricated in favor of half of every round
# epochs = 1001
num_rounds=8
seed = 1
log_interval = 400
log_test_value = 100
K = 5 #@param {type:"slider", min:5, max:50, step:1}
learning_rate = 1e-4 # Was 5e-4, which overtrained
discrete_data = True
alpha = 0 #@param [0, 1] {type:"raw"}
cuda = torch.cuda.is_available()

data_name = 'silhouettes' #@param['silhouettes','omniglot','freyfaces']

model_type = 'iwae' #@param['iwae','vrmax','vae']
torch.manual_seed(seed)

logging_filename = f'{model_type}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)

### silhouettes
# # Load data with random initialized train/test split
if os.environ.get('CLOUDSDK_CONFIG') is not None:
    fpath = "/content/drive/My Drive/data/caltech101_silhouettes_28.mat"
else:
    fpath = os.path.abspath('data/caltech101_silhouettes_28.mat')

data = loadmat(fpath)
data = 1-data.get('X')

np.random.seed(seed)
np.random.shuffle(data)

num_train = int(.9* data.shape[0])

data_train = data[:num_train]
data_test = data[num_train:]

data_train_t, data_test_t = T(data_train), T(data_test)

# Define likelihood functions
def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi))

def compute_log_probabitility_bernoulli(theta, obs, axis=1):
    return torch.sum(obs*torch.log(theta+1e-18) + (1-obs)*torch.log(1-theta+1e-18), axis)

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

    def compute_loss_for_batch(self, data, model, K=K,test=False):
        # data = (N,560)
        if model_type=='vae':
            alpha=1
        elif model_type in ('iwae','vrmax'):
            alpha=0
        else:
            # use whatever alpha is defined in hyperparameters
            if abs(alpha-1)<=1e-3:
                alpha=1

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
        elif model_type=='general_alpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True)

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, loss

def train(round_num,epoch):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        recon_batch, loss = model.compute_loss_for_batch(data, model)
        with detect_anomaly():
          loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Round number {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         round_num, epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))
        #     logging.info('Round number {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         round_num, epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    print('====> Round {}: Epoch: {} Average loss: {:.4f}'.format(
          round_num, epoch,  train_loss / len(train_loader.dataset)))
    logging.info('====> Round {}: Epoch: {} Average loss: {:.4f}'.format(
          round_num, epoch,  train_loss / len(train_loader.dataset)))

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(round_num,epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            _, loss = model.compute_loss_for_batch(data, model, K=5000, test=True)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,28),
                                      recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         f'{model_type}_{data_name}_K{K}_M{batch_size}/recons/reconstruction_' +str(round_num)+'_'+ str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Round {}: Epoch {}: Test set loss: {:.4f}'.format(round_num, epoch,test_loss))
    logging.info('====> Round {}: Epoch {}: Test set loss: {:.4f}'.format(round_num, epoch,test_loss))


# Initialize a model and data loaders
train_loader = DataLoader(TensorDataset(data_train_t), batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(TensorDataset(data_test_t), batch_size=test_batch_size, shuffle=True, pin_memory=True)

device = torch.device('cuda')
model = silhouettes_model().to(device)

# Call the training shenanigans
if torch.cuda.is_available():
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}', exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/samples', exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/recons', exist_ok=True)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for r in range(num_rounds + 1):
    current_round_lr = learning_rate * (10 ** (-r / num_rounds))
    optimizer = optim.Adam(model.parameters(), lr=current_round_lr)
    print(
        f"====== About to train for {2 ** r} epochs in round {r} with learning rate {round(current_round_lr, 7)}========")
    logging.info(
        f"====== About to train for {2 ** r} epochs in round {r} with learning rate {round(current_round_lr, 7)}========")
    for epoch in range(2 ** r):
        train(r, epoch)
    with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_LR0001.pt',
              'wb') as f:
        print(datetime.datetime.now())
        logging.info(datetime.datetime.now())
        torch.save(model, f)

    _test(r, epoch)
    with torch.no_grad():
        sample = torch.randn(64, 200).to(device)
        sample = model.decode(sample, test=True).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   f'{model_type}_{data_name}_K{K}_M{batch_size}/samples/sample_' + str(r) + '_' + str(epoch) + '.png')

_test(r, epoch)
with open(
        f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_LR0001_{r}_{epoch}.pt',
        'wb') as f:
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    torch.save(model, f)
print("Training finished")
logging.info("Training finished")