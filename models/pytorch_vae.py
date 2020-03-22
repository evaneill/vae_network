# THIS FILE IS NO LONGER KEPT UP TO DATE
# THIS FILE IS NO LONGER KEPT UP TO DATE
# THIS FILE IS NO LONGER KEPT UP TO DATE
# THIS FILE IS NO LONGER KEPT UP TO DATE
# THIS FILE IS NO LONGER KEPT UP TO DATE
# THIS FILE IS NO LONGER KEPT UP TO DATE


# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version
# SEE collab_pytorch_vae.py for up to date version
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
from utils import Loader
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=501, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--K', type=int, default=5, metavar='K',
                    help='how often to sample at each stochastic layer')
parser.add_argument('--model_type', type=str, default='iwae', metavar='model_type',
                    help='what mode (vae / iwae) to use')
parser.add_argument('--discrete_data', type=bool, default=True, metavar='discrete_data',
                    help='whether data is discrete or continuous')
parser.add_argument('--alpha', type=float, default=0, metavar='alpha',
                    help='alpha-value of Renyi divergence chosen')
parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                    help='set learning rate')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

test_batch_size = 32
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

use_freyfaces_and_not_mnist = False

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if use_freyfaces_and_not_mnist:
    my_loader = Loader('freyfaces')
    data = my_loader.load(output_type='tensor')
    data_train, data_test = data[0], data[1]
    train_loader = DataLoader(TensorDataset(data_train), batch_size=args.batch_size, **kwargs)
    test_loader = DataLoader(TensorDataset(data_test), batch_size=test_batch_size, **kwargs)

else:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=test_batch_size, shuffle=True, **kwargs)

assert args.model_type in ('vae', 'iwae')

class VAE1(nn.Module):
    def __init__(self):
        super(VAE1, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.K = args.K
    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        #eps = torch.randn_like(std)
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps*std

    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=args.K):
        # data = (B, F) = (B, 1, H, W)
        B, _, H, W = data.shape
        # if mode == 'vae':
        #     recon_batch, mu, logstd = model(data)
        #     return recon_batch, mu, logstd, loss_function(recon_batch, data, mu, logstd)
        # elif mode == 'iwae':
        # he wants (K, B, 784)
        # i have (B, 1, 28, 28)
        # data = data.view(B, H*W)
        # data = data.expand(args.K, B, H*W)
        # mu_h1, log_sigma_h1 = model.encode(data)
        # sigma_h1 = torch.exp(log_sigma_h1)
        # eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        #
        # #h1 = mu_h1 + sigma_h1 * eps
        # std = torch.exp(log_sigma_h1)
        # #eps = torch.randn_like(std)
        # h1 = mu_h1 + eps*std
        #
        # log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        # #log_Qh1Gx = torch.sum(-0.5 * eps** 2 - torch.log(sigma_h1), -1)
        #
        # p = model.decode(h1)
        # log_Ph1 = torch.sum(-0.5 * h1 ** 2, -1)
        # log_PxGh1 = torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p), -1)
        #
        # log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        # log_weight = log_weight - torch.max(log_weight, 0)[0]
        # weight = torch.exp(log_weight)
        # weight = weight / torch.sum(weight, 0)
        # weight = Variable(weight.data, requires_grad=False)
        # loss = -torch.mean(torch.sum(weight * (log_Ph1 + log_PxGh1 - log_Qh1Gx), 0))
        # return p, mu_h1, torch.log(sigma_h1), loss

        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)
        # log_q = torch.sum(-0.5 * ((z-mu)/torch.exp(logstd))**2 - logstd, 1) # - log(torch.sqrt(2*torch.pi))

        log_p_z = torch.sum(-0.5 * z ** 2, 1)
        # log_p_z = compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False), torch.zeros_like(z, requires_grad=False))
        recon = model.decode(z)
        if args.discrete_data:
            log_p = compute_log_probabitility_bernoulli(recon, data_k_vec)
            # log_p = torch.sum(data_k.view(-1, 784)*torch.log(decoded) + (1-data_k.view(-1, 784))*torch.log(1-decoded), 1)
            # log_p = F.binary_cross_entropy(decoded, data_k.view(-1, 784))
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p = compute_log_probabitility_gaussian(recon, data_k_vec, torch.zeros_like(recon))
            # log_p = torch.sum(-0.5 * (decoded-data_k.view(-1, 784))**2, 1)
        # hopefully this reshape operation magically works like always
        if args.model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif args.model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1)*1/K
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        # ws_norm = ws_matrix / torch.clamp(torch.sum(ws_matrix, 1, keepdim=True), 1e-9, np.inf)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return recon, 0, 0, loss

class VAE2(nn.Module):
    def __init__(self):
        super(VAE2, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = args.K
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

    def compute_loss_for_batch(self, data, model, K=args.K):
        # data = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = torch.sum(-0.5 * z ** 2, 1)
        decoded = model.decode(z)
        if args.discrete_data:
            log_p = compute_log_probabitility_bernoulli(decoded, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p = compute_log_probabitility_gaussian(decoded, data_k_vec, torch.zeros_like(decoded))
        # hopefully this reshape operation magically works like always
        if args.model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif args.model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1)*1/K
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, logstd, loss

class VAE3(nn.Module):
    # data_dim, q_dims, latents_dim, p_dims
    def __init__(self):
        super(VAE3, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 100)
        self.fc32 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc61 = nn.Linear(100, 50)
        self.fc62 = nn.Linear(100, 50)

        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc91 = nn.Linear(100, 100)
        self.fc92 = nn.Linear(100, 100)

        self.fc10 = nn.Linear(100, 200)
        self.fc11 = nn.Linear(200, 200)
        self.fc12 = nn.Linear(200, 784)

        self.K = args.K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu1, logstd1 = self.fc31(h2), self.fc32(h2)
        z1 = self.reparameterize(mu1, logstd1)

        h3 = torch.tanh(self.fc4(z1))
        h4 = torch.tanh(self.fc5(h3))
        mu2, logstd2 = self.fc61(h4), self.fc62(h4)
        z2 = self.reparameterize(mu2, logstd2)
        return z1, mu1, logstd1, z2, mu2, logstd2

    def decode_for_testing(self, z2):
        # h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z3))
        h4 = torch.tanh(self.fc11(h3))
        return torch.sigmoid(self.fc12(h4))

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z1, z2):
        #h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)

        #z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z1))
        h4 = torch.tanh(self.fc11(h3))
        return z1, mu3, logstd3, torch.sigmoid(self.fc12(h4))

        # # h3 = F.relu(self.fc3(z))
        # h1 = torch.tanh(self.fc7(z2))
        # h2 = torch.tanh(self.fc8(h1))
        # mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        # z3 = self.reparameterize(mu3, logstd3)
        #
        # h3 = torch.tanh(self.fc10(z3))
        # h4 = torch.tanh(self.fc11(h3))
        # return z3, mu3, logstd3, torch.sigmoid(self.fc12(h4))

    def forward(self, x):
        #mu, logstd = self.encode(x.view(-1, 784))
        z1, _, _, z2, mu, logstd = self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logstd)
        _, _, _, recon = self.decode(z1, z2)
        return recon, mu, logstd

    def compute_loss_for_batch(self, data, model, K=args.K):
        # he wants (K, B, 784)
        # i have (B, 1, 28, 28)
        # B, _, H, W = data.shape
        # data = data.view(B, H*W)
        # data = data.expand(args.K, B, H*W)
        # # mu_h1, log_sigma_h1 = model.encode(data)
        # # sigma_h1 = torch.exp(log_sigma_h1)
        # # eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        #
        # h1, mu_h1, log_sigma_h1, h2, mu_h2, log_sigma_h2 = self.encode(data)
        # sigma_h1, sigma_h2 = torch.exp(log_sigma_h1), torch.exp(log_sigma_h2)
        # log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        # log_Qh2Gh1 = torch.sum(-0.5*((h2-mu_h2)/sigma_h2)**2 - torch.log(sigma_h2), -1)
        #
        # #log_Qh1Gx = torch.sum(-0.5 * (eps1) ** 2 - torch.log(sigma_h1), -1)
        # #log_Qh2Gh1 = torch.sum(-0.5 * (eps2) ** 2 - torch.log(sigma_h2), -1)
        #
        # h1, mu_h1, sigma_h1, p = self.decode(h1, h2)
        # log_Ph2 = torch.sum(-0.5 * h2 ** 2, -1)
        # log_Ph1Gh2 = torch.sum(-0.5 * ((h1 - mu_h1) / sigma_h1) ** 2 - torch.log(sigma_h1), -1)
        # log_PxGh1 = torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p), -1)
        #
        # log_weight = log_Ph2 + log_Ph1Gh2 + log_PxGh1 - log_Qh1Gx - log_Qh2Gh1
        # log_weight = log_weight - torch.max(log_weight, 0)[0]
        # weight = torch.exp(log_weight)
        # weight = weight / torch.sum(weight, 0)
        # weight = Variable(weight.data, requires_grad=False)
        # loss = -torch.mean(torch.sum(weight * (log_Ph2 + log_Ph1Gh2 + log_PxGh1 - log_Qh1Gx - log_Qh2Gh1), 0))
        # return 0, 0, 0, loss


        # data = (B, F) = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        z1, mu1, logstd1, z2, mu2, logstd2 = model.encode(data_k_vec)
        # (B*K, #latents)
        log_q_1 = compute_log_probabitility_gaussian(z1, mu1, logstd1)
        log_q_2 = compute_log_probabitility_gaussian(z2, mu2, logstd2)

        log_p_z = torch.sum(-0.5 * z2 ** 2, 1)
        # log_p_z = compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False), torch.zeros_like(z, requires_grad=False))
        z3, mu3, logstd3, recon = model.decode(z1, z2)
        log_p_1 = compute_log_probabitility_gaussian(z3, mu3, logstd3)
        if args.discrete_data:
            log_p_2 = compute_log_probabitility_bernoulli(recon, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p_2 = compute_log_probabitility_gaussian(recon, data_k_vec, torch.zeros_like(recon))
            # log_p = torch.sum(-0.5 * (decoded-data_k.view(-1, 784))**2, 1)
        # hopefully this reshape operation magically works like always
        if args.model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B, K)
        elif args.model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B*K, 1)*1/K
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        # ws_norm = ws_matrix / torch.clamp(torch.sum(ws_matrix, 1, keepdim=True), 1e-9, np.inf)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return recon, 0, 0, loss

class VAE4(nn.Module):
    def __init__(self):
        super(VAE4, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = args.K
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

    def compute_loss_for_batch(self, data, model, K=args.K):
        # data = (B, F) = (B, 1, H, W)
        B, _, H, W = data.shape
        alpha = args.alpha
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)
        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)
        log_p_z = torch.sum(-0.5 * z ** 2, 1)
        decoded = model.decode(z)
        if args.discrete_data:
            log_p = compute_log_probabitility_bernoulli(decoded, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p = compute_log_probabitility_gaussian(decoded, data_k_vec, torch.zeros_like(decoded))
        # hopefully this reshape operation magically works like always
        if args.model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif args.model_type == 'vae':
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1) * 1 / K
        if alpha != 1:
            log_w_matrix_alpha = log_w_matrix * (1-alpha)
        else:
            log_w_matrix_alpha = log_w_matrix

        log_w_minus_max = log_w_matrix_alpha - torch.max(log_w_matrix_alpha, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        ws_sum_per_datapoint = torch.sum(log_w_matrix_alpha * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)
        if alpha != 1:
            loss *= 1/(1-alpha)
        return decoded, mu, logstd, loss

class VAE5(nn.Module):
    # data_dim, q_dims, latents_dim, p_dims
    def __init__(self):
        super(VAE5, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 100)
        self.fc32 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc61 = nn.Linear(100, 50)
        self.fc62 = nn.Linear(100, 50)

        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc91 = nn.Linear(100, 100)
        self.fc92 = nn.Linear(100, 100)

        self.fc10 = nn.Linear(100, 200)
        self.fc11 = nn.Linear(200, 200)
        self.fc12 = nn.Linear(200, 784)

        self.K = args.K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu1, logstd1 = self.fc31(h2), self.fc32(h2)
        z1 = self.reparameterize(mu1, logstd1)

        h3 = torch.tanh(self.fc4(z1))
        h4 = torch.tanh(self.fc5(h3))
        mu2, logstd2 = self.fc61(h4), self.fc62(h4)
        z2 = self.reparameterize(mu2, logstd2)
        return z1, mu1, logstd1, z2, mu2, logstd2

    def decode_for_testing(self, z2):
        # h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z3))
        h4 = torch.tanh(self.fc11(h3))
        return torch.sigmoid(self.fc12(h4))

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z1, z2):
        #h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        #z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z1))
        h4 = torch.tanh(self.fc11(h3))
        return z1, mu3, logstd3, torch.sigmoid(self.fc12(h4))

        # # h3 = F.relu(self.fc3(z))
        # h1 = torch.tanh(self.fc7(z2))
        # h2 = torch.tanh(self.fc8(h1))
        # mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        # z3 = self.reparameterize(mu3, logstd3)
        #
        # h3 = torch.tanh(self.fc10(z3))
        # h4 = torch.tanh(self.fc11(h3))
        # return z3, mu3, logstd3, torch.sigmoid(self.fc12(h4))

    def forward(self, x):
        #mu, logstd = self.encode(x.view(-1, 784))
        z1, _, _, z2, mu, logstd = self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logstd)
        _, _, _, recon = self.decode(z1, z2)
        return recon, mu, logstd

    def compute_loss_for_batch(self, data, model, K=args.K):
        # he wants (K, B, 784)
        # i have (B, 1, 28, 28)
        # B, _, H, W = data.shape
        # data = data.view(B, H*W)
        # data = data.expand(args.K, B, H*W)
        # # mu_h1, log_sigma_h1 = model.encode(data)
        # # sigma_h1 = torch.exp(log_sigma_h1)
        # # eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        #
        # h1, mu_h1, log_sigma_h1, h2, mu_h2, log_sigma_h2 = self.encode(data)
        # sigma_h1, sigma_h2 = torch.exp(log_sigma_h1), torch.exp(log_sigma_h2)
        # log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        # log_Qh2Gh1 = torch.sum(-0.5*((h2-mu_h2)/sigma_h2)**2 - torch.log(sigma_h2), -1)
        #
        # #log_Qh1Gx = torch.sum(-0.5 * (eps1) ** 2 - torch.log(sigma_h1), -1)
        # #log_Qh2Gh1 = torch.sum(-0.5 * (eps2) ** 2 - torch.log(sigma_h2), -1)
        #
        # h1, mu_h1, sigma_h1, p = self.decode(h1, h2)
        # log_Ph2 = torch.sum(-0.5 * h2 ** 2, -1)
        # log_Ph1Gh2 = torch.sum(-0.5 * ((h1 - mu_h1) / sigma_h1) ** 2 - torch.log(sigma_h1), -1)
        # log_PxGh1 = torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p), -1)
        #
        # log_weight = log_Ph2 + log_Ph1Gh2 + log_PxGh1 - log_Qh1Gx - log_Qh2Gh1
        # log_weight = log_weight - torch.max(log_weight, 0)[0]
        # weight = torch.exp(log_weight)
        # weight = weight / torch.sum(weight, 0)
        # weight = Variable(weight.data, requires_grad=False)
        # loss = -torch.mean(torch.sum(weight * (log_Ph2 + log_Ph1Gh2 + log_PxGh1 - log_Qh1Gx - log_Qh2Gh1), 0))
        # return 0, 0, 0, loss

        alpha = args.alpha
        # data = (B, F) = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        z1, mu1, logstd1, z2, mu2, logstd2 = model.encode(data_k_vec)
        # (B*K, #latents)
        log_q_1 = compute_log_probabitility_gaussian(z1, mu1, logstd1)
        log_q_2 = compute_log_probabitility_gaussian(z2, mu2, logstd2)

        log_p_z = torch.sum(-0.5 * z2 ** 2, 1)
        # log_p_z = compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False), torch.zeros_like(z, requires_grad=False))
        z3, mu3, logstd3, recon = model.decode(z1, z2)
        log_p_1 = compute_log_probabitility_gaussian(z3, mu3, logstd3)
        if args.discrete_data:
            log_p_2 = compute_log_probabitility_bernoulli(recon, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p_2 = compute_log_probabitility_gaussian(recon, data_k_vec, torch.zeros_like(recon))
            # log_p = torch.sum(-0.5 * (decoded-data_k.view(-1, 784))**2, 1)
        # hopefully this reshape operation magically works like always
        if args.model_type == 'iwae':
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B, K)
        elif args.model_type == 'vae':
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B*K, 1) * 1 / K
        if alpha != 1:
            log_w_matrix_alpha = log_w_matrix * (1-alpha)
        else:
            log_w_matrix_alpha = log_w_matrix
        log_w_minus_max = log_w_matrix_alpha - torch.max(log_w_matrix_alpha, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        # ws_norm = ws_matrix / torch.clamp(torch.sum(ws_matrix, 1, keepdim=True), 1e-9, np.inf)
        ws_sum_per_datapoint = torch.sum(log_w_matrix_alpha * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)
        if alpha != 1:
            loss *= 1/(1-alpha)
        return recon, 0, 0, loss

# 1 = vanilla Pytorch VAE, VAE/IWAE loss
# 2 = 1 stochastic layer from IWAE, VAE/IWAE loss
# 3 = 2 stochastic layers from IWAE, VAE/IWAE loss
# 4 = 1 stochastic layer from IWAE, renyi loss
# 5 = 2 stochastic layers from IWAE, renyi loss
model = VAE2().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logstd):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
    return BCE + KLD

def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)

def compute_log_probabitility_bernoulli(obs, p, axis=1):
    return torch.sum(p*torch.log(obs) + (1-p)*torch.log(1-obs), axis)


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
        if batch_idx % args.log_interval == 0:
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
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            _, _, _, loss = model.compute_loss_for_batch(data, model, 5000)
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

if __name__ == "__main__":
    if torch.cuda.is_available(): print("Training on GPU")
    print(datetime.datetime.now())
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 500 == 1:
            _test(epoch)
            with torch.no_grad():
                if isinstance(model, VAE1):
                    sample = torch.randn(64, 20).to(device)
                else:
                    sample = torch.randn(64, 50).to(device)
                if not isinstance(model, VAE3) and not isinstance(model, VAE5):
                    sample = model.decode(sample).cpu()
                else:
                    z2 = torch.randn(64, 50).to(device)
                    sample = model.decode_for_testing(z2).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           'results/sample_' + str(epoch) + '.png')
    print(datetime.datetime.now())
    print("Training finished")