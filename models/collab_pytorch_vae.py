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

batch_size = 128
test_batch_size = 32
epochs = 501
seed = 1
log_interval = 100
K = 5
learning_rate = 5e-4
discrete_data = False
alpha = 0
cuda = torch.cuda.is_available()

model_type = 'iwae'
torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


if os.environ.get('CLOUDSDK_CONFIG') is not None:   
    fpath = "/content/drive/My Drive/data/freyfaces.pkl"
else:
    fpath = os.path.abspath('data/freyfaces.pkl')

f = open(fpath,'rb')
data = pickle.load(f,encoding='latin1')
f.close()

np.random.seed(seed)
np.random.shuffle(data)

train_ratio = .9
num_train = int(train_ratio* data.shape[0])

data_train_t = T(1-data[:num_train])
data_test_t = T(1-data[num_train:])

assert model_type in ('vae', 'iwae')

class freyface_model(nn.Module):
    def __init__(self):
        super(freyface_model, self).__init__()

        self.fc1 = nn.Linear(560, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 20)
        self.fc32 = nn.Linear(200, 20)

        self.fc4 = nn.Linear(20, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 560)
        self.fc7 = nn.Linear(200, 560)

        self.K = K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        h3 = F.softplus(self.fc4(z))
        h4 = F.softplus(self.fc5(h3))
        return self.fc6(h4), self.fc7(h4) #mu, log_sigma

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 560))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K):
        # data = (N,560)
        data_k_vec = data.repeat_interleave(K,0)

        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = torch.sum(-0.5 * z ** 2, 1)
        decoded = model.decode(z) # decoded = (pmu, plog_sigma)
        pmu, plog_sigma = decoded
        log_p = compute_log_probabitility_gaussian(data_k_vec,pmu, plog_sigma)
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1)*1/K
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, logstd, loss

model = freyface_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True)


def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi)) 

def compute_log_probabitility_bernoulli(obs, p, axis=1):
    return torch.sum(p*torch.log(obs) + (1-p)*torch.log(1-obs), axis)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        (recon_batch,_), _, _, loss = model.compute_loss_for_batch(data, model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch,  train_loss / len(train_loader.dataset)))

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(test_loader):
            data = data.to(device)
            (recon_batch,_), mu, logvar = model(data)
            _, _, _, loss = model.compute_loss_for_batch(data, model, K=5000)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,20),
                                      recon_batch.view(test_batch_size, 1, 28, 20)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    if torch.cuda.is_available(): print("Training on GPU")
    os.makedirs('results', exist_ok=True)
    print(datetime.datetime.now())
    for epoch in range(1, epochs + 1):
        train(epoch)
        if epoch % 500 == 1:
            _test(epoch)
            with torch.no_grad():
                z2 = torch.randn(64, 20).to(device)
                (sample,_) = model.decode(z2)
                sample = sample.cpu()
                save_image(sample.view(64, 1, 28, 20),
                            'results/sample_' + str(epoch) + '.png')
    print(datetime.datetime.now())
    print("Training finished")