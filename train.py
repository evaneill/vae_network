from models.loss import VRBound
from models.network import VRalphaNet

from utils import Loader, DATASETS
import argparse

import math

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image

import os
from datetime import datetime

def main(data,model,alpha,optimize_on,batch_size,learning_rate,n_epochs,imsize,test_batch_size,eval_log_marginal=True,log_interval=10):

	data_train, data_test = data[0], data[1]

	data_train_loader = DataLoader(TensorDataset(data_train),batch_size = batch_size)
	data_test_loader = DataLoader(TensorDataset(data_test),batch_size = test_batch_size)

	if torch.cuda.is_available():
		print("Training on GPU")
		device = torch.device('cuda')
	else:
		print("Training on CPU")
		device = torch.device('cpu')

	# model.encoder = model.encoder.to(device)
	# model.decoder = model.decoder.to(device)
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	test_loss_record = []
	print(datetime.now())
	for epoch in range(1,n_epochs+1):
		train(epoch,model,optimizer,alpha,data_train_loader,optimize_on,device,imsize)
		if (epoch-1) % log_interval==0:
			test_loss = test(epoch,model,0,data_test_loader,'full_bound',device,imsize,K=5000)
			test_loss_record.append(test_loss)

	print("finished training")
	print(datetime.now())
	return test_loss_record

def train(epoch,model,optimizer,alpha,data_loader,optimize_on,device,imsize):
	model.train()
	train_loss = 0
	(H,W) = imsize
	for batch_idx, [data] in enumerate(data_loader):
		data = data.to(device)
		optimizer.zero_grad()
		output, q_mu, q_log_sigma = model.forward(data)
		# print(f"Call in TRAIN() using K={model.encoder.K}")
		loss = -1*VRBound(alpha,model,output,q_mu, q_log_sigma,optimize_on=optimize_on)
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

def test(epoch,model,alpha,data_loader,optimize_on,device,imsize,K = None):
	if K is None:
		K = model.encoder.K

	model.eval()
	test_loss = 0
	(H,W) = imsize
	with torch.no_grad():
	    for i, [data] in enumerate(data_loader):
	        data = data.to(device)
	        output, q_mu, q_log_sigma = model.forward(data,K = K)
	        # print(f"Call in TEST() using K={K}")
	        test_loss += -1*VRBound(alpha,model,output,q_mu, q_log_sigma,K=5000,optimize_on=optimize_on)
	        recon_batch = model.get_recon(data)
	        if i == 0:
	            n = min(data.size(0), 8)
	            comparison = torch.cat([data[:n].view(-1,1,H,W),
	                                  recon_batch.view(-1, 1, H, W)[:n]])
	            save_image(comparison.cpu(),
	                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

	test_loss /= len(data_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss 

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='try to train a model!')

	# Model architecture
	parser.add_argument('--L','-l',type=int,choices=[1,2],default=1)

	# Data to fit. Will assume freyfaces is "continuous" data type, and the rest binary
	parser.add_argument('--data',choices=DATASETS.keys())

	# Loss choice & training parameters
	parser.add_argument('--alpha','-a',type=float,default=1.)#,description='alpha of Renyi Alpha-Divergence used in deriving bound')
	parser.add_argument('--K','-k',type=int,default=5)#,description='number of samples drawn per input observation')
	parser.add_argument('--batch_size','-batch',type=int,default=100)#,description='batch size of training passes (before sampling)')
	parser.add_argument('--test_batch_size','-testbatch',type=int,default=20) # batch size for evaluation of runtime test error and K=5000 test
	parser.add_argument('--optimize_on',choices=['full_bound','max','sample'],default='full_bound')#,description='whether a full sum should be taken, or only the max-weight observation when estimating the bound')
	parser.add_argument('--learning_rate','-lr',type=float,default=-1.)
	parser.add_argument('--n_epoch','-e',type=int,default=100)

	# Logging
	parser.add_argument('--log_interval','-log',type=int,default=10)

	args = parser.parse_args()
	data_name = args.data
	L = args.L

	alpha = args.alpha
	K = args.K
	batch_size = args.batch_size
	test_batch_size = args.test_batch_size
	optimize_on = args.optimize_on
	learning_rate = args.learning_rate
	n_epochs = args.n_epoch

	log_interval = args.log_interval

	if data_name=="freyfaces":
		imsize=(28,20)
		data_type="continuous"
		if learning_rate<0:
			# That used in paper
			learning_rate = 5e-4
	else:
		data_type="binary"
		imsize=(28,28)
		if learning_rate<0:
			# That used in paper, though exact training scheme won't be implemented here (multiple epochs w/ changing LR)
			learning_rate = 1e-3

	my_loader = Loader(data_name)
	data = my_loader.load(output_type='tensor')

	# os.makedirs(f"{data_name}_alpha={round(alpha,4)}_{optimize_on}_K{K}")
	os.makedirs('results/',exist_ok=True)
	# This will use the same architectures 
	if L==1:
		if data_name=='freyfaces':
			model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(20,'s')],'softplus',data_type=data_type,K=K)
		elif data_name=='silhouettes':
			model =  VRAlphaNet(data[0],[(500,'d'),(200,'s')],'tanh',data_type=data_type,K=K)
		else:
			model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(50,'s')],'tanh',data_type=data_type,K=K)
	elif L==2:
		model = VRalphaNet(data[0],[(200,'d'),(200,'d'),(100,'s'),(100,'d'),(100,'d'),(50,'s')],'tanh',data_type=data_type,K=K)

	main(data,model,alpha,optimize_on,batch_size,learning_rate,n_epochs,imsize,test_batch_size,log_interval=log_interval)