import datetime
import logging
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim, Tensor as T
from torch.autograd import detect_anomaly
from torch.distributions.multinomial import Multinomial
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.makedirs('results', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# Hyperparameters
alpha = -500 # alpha value used in Renyi alpha-divergence, ignored when model_type is vae/iwae/vrmax
K = 5 # number of samples taken per input data point
L = 2 # number of stochastic layers in network architecture; either 1 or 2

model_type = 'vralpha' # one of ['vae', 'iwae', 'vrmax', 'vralpha', 'general_alpha']
data_name = 'mnist' # one of ['mnist', 'fashion', 'fashionmnist']

epochs = 501
learning_rate = 1e-3

log_interval = 1 # how frequently to log average training loss
test_interval = 1 # how frequently to test
train_batch_size = 100 # batch size during training
test_batch_size = 32 # batch size used during testing, different than training because testing is done with K=5000

seed = 1 # fixed seed
torch.manual_seed(seed)
discrete_data = True # always True for MNIST/FashionMNIST, but needed when training on continuous data

assert(L in [1, 2]) # we only have networks with 1 or 2 stochastic layers
assert(model_type in ['vae', 'iwae', 'vrmax', 'vralpha', 'general_alpha'])
assert(not(alpha==1 and model_type in ['vralpha', 'general_alpha'])) # divide by 0 error otherwise
assert(data_name in ['mnist', 'fashion', 'fashionmnist'])

class mnist_omniglot_model1(nn.Module):
    def __init__(self, alpha):
        super(mnist_omniglot_model1, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = K
        self.alpha = alpha

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, test=False):
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
        if model_type == 'iwae' or test:
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1)*1/K
        elif model_type=='general_alpha' or model_type=='vralpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type == 'vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1, keepdim=True).values
            return -torch.sum(log_w_matrix)
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

        return loss

# Define the model
class mnist_omniglot_model2(nn.Module):
    def __init__(self, alpha):
        super(mnist_omniglot_model2, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 100)  # stochastic 1
        self.fc32 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc61 = nn.Linear(100, 50)  # Innermost (stochastic 2)
        self.fc62 = nn.Linear(100, 50)

        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc81 = nn.Linear(100, 100)  # stochastic 1
        self.fc82 = nn.Linear(100, 100)

        self.fc9 = nn.Linear(100, 200)
        self.fc10 = nn.Linear(200, 200)
        self.fc11 = nn.Linear(200, 784)  # reconstruction

        self.K = K
        self.alpha = alpha

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu, log_std = self.fc31(h2), self.fc32(h2)

        z1 = self.reparameterize(mu, log_std)
        h3 = torch.tanh(self.fc4(z1))
        h4 = torch.tanh(self.fc5(h3))

        return self.fc61(h4), self.fc62(h4), [x, z1]

    def reparameterize(self, mu, logstd, test=False):
        std = torch.exp(logstd)
        if test == True:
            eps = torch.zeros_like(mu)
        else:
            eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, test=False):
        h5 = torch.tanh(self.fc7(z))
        h6 = torch.tanh(self.fc8(h5))
        mu, log_std = self.fc81(h6), self.fc82(h6)

        z1 = self.reparameterize(mu, log_std, test=test)
        h7 = torch.tanh(self.fc9(z1))
        h8 = torch.tanh(self.fc10(h7))

        return torch.sigmoid(self.fc11(h8))

    def forward(self, x):
        mu, logstd, _ = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, test=False):
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H * W)
        # data_k_vec = data.repeat_interleave(K, 0)

        mu, log_std, [x, z1] = self.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, log_std)

        # Log p(z) (prior)
        log_p_z = torch.sum(-0.5 * z ** 2, 1) - .5 * z.shape[1] * T.log(torch.tensor(2 * np.pi))

        # q (z | h1)
        log_qz_h1 = compute_log_probabitility_gaussian(z, mu, log_std)

        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu, log_std = self.fc31(h2), self.fc32(h2)

        # q (h1 | x)
        log_qh1_x = compute_log_probabitility_gaussian(z1, mu, log_std)

        h5 = torch.tanh(self.fc7(z))
        h6 = torch.tanh(self.fc8(h5))
        mu, log_std = self.fc81(h6), self.fc82(h6)

        # log p(h1 | z)
        log_ph1_z = compute_log_probabitility_gaussian(z1, mu, log_std)

        h7 = torch.tanh(self.fc9(z1))
        h8 = torch.tanh(self.fc10(h7))

        decoded = torch.sigmoid(self.fc11(h8))
        # log p(x | h1)
        if discrete_data:
            log_px_h1 = compute_log_probabitility_bernoulli(decoded, x)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_px_h1 = compute_log_probabitility_gaussian(decoded, x, torch.zeros_like(decoded))

        # log_px_h1 = compute_log_probabitility_bernoulli(decoded, x)

        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or test == True:
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K)
        elif model_type == 'vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, 1) * 1 / K
            return -torch.sum(log_w_matrix)
        elif model_type == 'general_alpha' or model_type == 'vralpha':
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K) * (1 - self.alpha)
        elif model_type == 'vrmax':
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K).max(axis=1,
                                                                                                     keepdim=True).values
            return -torch.sum(log_w_matrix)

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        if model_type == 'vralpha' and not test:
            sample_dist = Multinomial(1, ws_norm)
            ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
        else:
            ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)

        if model_type in ["general_alpha", "vralpha"] and not test:
            ws_sum_per_datapoint /= (1 - self.alpha)
        loss = -torch.sum(ws_sum_per_datapoint)

        return loss

# Compute N(obs| mu, sigma) for all K samples and sum over probabilities of the K samples
def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi))

# Compute Ber(obs| theta) for all K samples and sum over probabilities of the K samples
def compute_log_probabitility_bernoulli(theta, obs, axis=1):
    # 1e-18 needed to avoid numerical errors
    return torch.sum(obs*torch.log(theta+1e-18) + (1-obs)*torch.log(1-theta+1e-18), axis)

# Define the model
# class silhouettes_model(nn.Module):
#     def __init__(self):
#         super(silhouettes_model, self).__init__()
#
#         self.fc1 = nn.Linear(784, 500)
#         self.fc21 = nn.Linear(500, 200)
#         self.fc22 = nn.Linear(500, 200)
#
#         self.fc3 = nn.Linear(200, 500)
#         self.fc4 = nn.Linear(500, 784)
#
#         self.K = K
#         self.alpha = alpha
#
#     def encode(self, x):
#         h1 = torch.tanh(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)
#
#     def reparameterize(self, mu, logstd,test=False):
#         std = torch.exp(logstd)
#         if test==True:
#           eps = torch.zeros_like(mu)
#         else:
#           eps = torch.randn_like(std)
#         return mu + eps*std
#
#     def decode(self, z,test=False):
#         h3 = torch.tanh(self.fc3(z))
#
#         return torch.sigmoid(self.fc4(h3))
#
#     def forward(self, x):
#         mu, logstd = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logstd)
#         return self.decode(z), mu, logstd
#
#     def compute_loss_for_batch(self, data, model, K=K,test=False):
#         alpha = self.alpha
#         # data = (N,560)
#         if model_type=='vae':
#             alpha=1
#         elif model_type in ('iwae','vrmax'):
#             alpha=0
#         else:
#             # use whatever alpha is defined in hyperparameters
#             if abs(alpha-1)<=1e-3:
#                 alpha=1
#
#         data_k_vec = data.repeat_interleave(K,0)
#
#         mu, logstd = model.encode(data_k_vec)
#         # (B*K, #latents)
#         z = model.reparameterize(mu, logstd)
#
#         # summing over latents due to independence assumption
#         # (B*K)
#         log_q = compute_log_probabitility_gaussian(z, mu, logstd)
#
#         log_p_z = torch.sum(-0.5 * z ** 2, 1)-.5*z.shape[1]*T.log(torch.tensor(2*np.pi))
#         decoded = model.decode(z) # decoded = (pmu, plog_sigma)
#         log_p = compute_log_probabitility_bernoulli(decoded,data_k_vec)
#         # hopefully this reshape operation magically works like always
#         if model_type == 'iwae' or test:
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
#         elif model_type =='vae':
#             # treat each sample for a given data point as you would treat all samples in the minibatch
#             # 1/K value because loss values seemed off otherwise
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1)*1/K
#         elif model_type=='general_alpha' or model_type=='vralpha':
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
#         elif model_type=='vrmax':
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True).values
#
#         log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
#         ws_matrix = torch.exp(log_w_minus_max)
#         ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
#
#         if model_type == 'vralpha' and not test:
#             sample_dist = Multinomial(1, ws_norm)
#             ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
#         else:
#             ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
#
#         if model_type in ["general_alpha", "vralpha"] and not test:
#             ws_sum_per_datapoint /= (1 - alpha)
#         loss = -torch.sum(ws_sum_per_datapoint)
#
#         return decoded, loss
#
# # Define the model
# class freyface_model(nn.Module):
#     def __init__(self, alpha):
#         super(freyface_model, self).__init__()
#
#         self.fc1 = nn.Linear(560, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc31 = nn.Linear(200, 20)
#         self.fc32 = nn.Linear(200, 20)
#
#         self.fc4 = nn.Linear(20, 200)
#         self.fc5 = nn.Linear(200, 200)
#         self.fc6 = nn.Linear(200, 560)
#         self.fc7 = nn.Linear(200, 560)
#
#         self.K = K
#         self.alpha = alpha
#
#     def encode(self, x):
#         h1 = F.softplus(self.fc1(x))
#         h2 = F.softplus(self.fc2(h1))
#         return self.fc31(h2), self.fc32(h2)
#
#     def reparameterize(self, mu, logstd):
#         std = torch.exp(logstd)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z, test=False):
#         h3 = F.softplus(self.fc4(z))
#         h4 = F.softplus(self.fc5(h3))
#         return self.fc6(h4), self.fc7(h4)  # mu, log_sigma
#
#     def forward(self, x):
#         mu, logstd = self.encode(x.view(-1, 560))
#         z = self.reparameterize(mu, logstd)
#         return self.decode(z), mu, logstd
#
#     def compute_loss_for_batch(self, data, model, K=K, test=False):
#         alpha = self.alpha
#         # data = (N,560)
#         if model_type == 'vae':
#             alpha = 1
#         elif model_type in ('iwae', 'vrmax'):
#             alpha = 0
#         else:
#             # use whatever alpha is defined in hyperparameters
#             if abs(alpha - 1) <= 1e-3:
#                 alpha = 1
#
#         data_k_vec = data.repeat_interleave(K, 0)
#
#         mu, logstd = model.encode(data_k_vec)
#         # (B*K, #latents)
#         z = model.reparameterize(mu, logstd)
#
#         # summing over latents due to independence assumption
#         # (B*K)
#         log_q = compute_log_probabitility_gaussian(z, mu, logstd)
#
#         log_p_z = torch.sum(-0.5 * z ** 2, 1) - .5 * z.shape[1] * T.log(torch.tensor(2 * np.pi))
#         decoded = model.decode(z)
#         (pmu, plog_sigma) = decoded
#         log_p = compute_log_probabitility_gaussian(data_k_vec, pmu, plog_sigma)
#         # hopefully this reshape operation magically works like always
#         if model_type == 'iwae' or test == True:
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K)
#         elif model_type == 'vae':
#             # treat each sample for a given data point as you would treat all samples in the minibatch
#             # 1/K value because loss values seemed off otherwise
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, 1) * 1 / K
#         elif model_type == 'general_alpha' or model_type == 'vralpha':
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)
#         elif model_type == 'vrmax':
#             log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1, keepdim=True).values
#
#         log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
#         ws_matrix = torch.exp(log_w_minus_max)
#         ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
#
#         if model_type == 'vralpha' and not test:
#             sample_dist = Multinomial(1, ws_norm)
#             ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
#         else:
#             ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
#
#         if model_type in ["general_alpha", "vralpha"] and not test:
#             ws_sum_per_datapoint /= (1 - alpha)
#         loss = -torch.sum(ws_sum_per_datapoint)
#
#         return decoded, loss


# train and test functions
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g. (128, 1, 28, 28) for MNIST with B=128)
        data = data.to(device)
        optimizer.zero_grad()

        loss = model.compute_loss_for_batch(data, model)
        with detect_anomaly():
            loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if epoch % log_interval == 0:
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
        logging.info(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.compute_loss_for_batch(data, model, K=5000,test=True)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         f'results/reconstruction_{model_type}_L={L}_{data_name}_alpha={alpha}_K={K}_epoch={epoch}.png',
                           nrow=n)
                noise = torch.randn(64, 50).to(device)
                sample = model.decode(noise).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           f'results/sample_{model_type}_L={L}_{data_name}_alpha={alpha}_K={K}_epoch={epoch}.png')
    test_loss /= len(test_loader.dataset)
    print(f'====> Epoch: {epoch} Test set loss: {test_loss:.4f}')
    logging.info(f'====> Epoch: {epoch} Test set loss: {test_loss:.4f}')
    return test_loss

def load_data_and_initialize_loaders(data_name, train_batch, test_batch):
    data_name = data_name.lower()

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if data_name == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    elif data_name == 'fashion' or data_name == 'fashionmnist':
        train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor())
    elif data_name == 'omniglot':
        train_data = datasets.Omniglot(root='./data', background=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.Omniglot(root='./data', background=False, download=True, transform=transforms.ToTensor())
    # else: raise Exception("Data name not recognized")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_batch, shuffle = True, ** kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = test_batch, shuffle = True, ** kwargs)
    return train_loader, test_loader

if __name__ == "__main__":
    if L==1:
        model = mnist_omniglot_model1(alpha).to(device)
    else:
        model = mnist_omniglot_model2(alpha).to(device)
    train_loader, test_loader = load_data_and_initialize_loaders(data_name, train_batch_size, test_batch_size)
    if torch.cuda.is_available():
        print("Training on GPU")
        logging.info("Training on GPU")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'{datetime.datetime.now()} \nStarting training')
    logging.info(f'{datetime.datetime.now()} \nStarting training')
    for e in range(1, epochs+1):
        train(e)
        if e % test_interval == 0:
            _test(e)
    _test(epochs)
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    print("Training finished")
    logging.info("Training finished")
    print("Saving model")
    torch.save(model.state_dict(),
               f'models/{model_type}_L={L}_{data_name}_alpha={alpha}_K={K}_epochs={epochs}.pt')