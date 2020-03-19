from models.loss import VRBound
from models.network import VRalphaNet

from utils import Loader, DATASETS
import argparse

import math

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image

def main(data,model,alpha,optimize_on,batch_size,learning_rate,n_epochs):

	data_train, data_test = data[0], data[1]

	data_train_loader = DataLoader(TensorDataset(data_train),batch_size = batch_size)
	data_test_loader = DataLoader(TensorDataset(data_test),batch_size = batch_size)

	device = torch.device('cpu')

	# model.encoder = model.encoder.to(device)
	# model.decoder = model.decoder.to(device)
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(1,n_epochs+1):
		train(epoch,model,optimizer,alpha,data_train_loader,optimize_on,device)
		test(epoch,model,optimizer,alpha,data_test_loader,optimize_on,device)


def train(epoch,model,optimizer,alpha,data_loader,optimize_on,device):
	model.train()
	train_loss = 0
	for batch_idx, [data] in enumerate(data_loader):
		data = data.to(device)
		optimizer.zero_grad()
		output, q_mu, q_log_sigmasq = model.forward(data)
		loss = -1*VRBound(alpha,model,output,q_mu, q_log_sigmasq,optimize_on=optimize_on)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % 60 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			    epoch, batch_idx * len(data), len(data_loader.dataset),
			    100. * batch_idx / len(data_loader),
			    loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
	      epoch, train_loss / len(data_loader.dataset)))

def test(epoch,model,optimizer,alpha,data_loader,optimize_on,device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(data_loader):
            data = data.to(device)
            output, q_mu, q_log_sigmasq = model.forward(data)
            test_loss += -1*VRBound(alpha,model,output,q_mu, q_log_sigmasq,optimize_on=optimize_on)
            recon_batch = model.get_recon(data)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,28),
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='try to train a model!')

	# Model architecture
	parser.add_argument('--L','-l',type=int,choices=[1,2],default=1)

	# Data to fit. Will assume freyfaces is "continuous" data type, and the rest binary
	parser.add_argument('--data',choices=DATASETS.keys())

	# Loss choice & training parameters
	parser.add_argument('--alpha','-a',type=float,default=1.)#,description='alpha of Renyi Alpha-Divergence used in deriving bound')
	parser.add_argument('--n_samples','-k',type=int,default=5)#,description='number of samples drawn per input observation')
	parser.add_argument('--batch_size','-batch',type=int,default=100)#,description='batch size of training passes (before sampling)')
	parser.add_argument('--optimize_on',choices=['full_lowerbound','max'],default='full_lowerbound')#,description='whether a full sum should be taken, or only the max-weight observation when estimating the bound')
	parser.add_argument('--learning_rate','-lr',type=float,default=-1.)
	parser.add_argument('--n_epoch','-e',type=float,default=100)

	args = parser.parse_args()
	data_name = args.data
	L = args.L

	alpha = args.alpha
	n_samples = args.n_samples
	batch_size = args.batch_size
	optimize_on = args.optimize_on
	learning_rate = args.learning_rate
	n_epochs = args.n_epoch

	if data_name=="freyfaces":
		data_type="continuous"
		if learning_rate<0:
			# That used in paper
			learning_rate = 5e-4
	else:
		data_type="binary"
		if learning_rate<0:
			# That used in paper, though exact training scheme won't be implemented here (multiple epochs w/ changing hyperparameters)
			learning_rate = 1e-3

	my_loader = Loader(data_name)
	data = my_loader.load(output_type='tensor')

	# This will use the same architectures 
	if L==1:
		if data_name=='freyfaces':
			model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(20,'s')],'softplus',data_type=data_type,n_samples=n_samples)
		elif data_name=='silhouettes':
			model =  VRAlphaNet(data[0],[(500,'d'),(200,'s')],'tanh',data_type=data_type,n_samples = n_samples)
		else:
			model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(50,'s')],'tanh',data_type=data_type,n_samples=n_samples)
	elif L==2:
		model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(100,'s'),(100,'d'),(100,'d'),(50,'s')],'tanh',data_type=data_type,n_samples=n_samples)

	main(data,model,alpha,optimize_on,batch_size,learning_rate,n_epochs)