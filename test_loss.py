import argparse
from utils import Loader, DATASETS

from torch import Tensor as T

from models import network as net
from models.loss import renyiDivergence

def main(data,model,alpha):

	print("Turning into tensors...")
	data_train,data_test = data[0], data[1]
	data_train_t, data_test_t = T(data_train), T(data_test)

	print("Trying to feed forward...")
	output, qmu, pmu, qlog_sigmasq, plog_sigmasq = model.forward(data_train_t[:10,:])

	print(f"Trying to test loss with alpha={alpha} and L={len(model.encoder.layers)}")
	print(f"==== Loss = {renyiDivergence(alpha,model,output)} =====")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='test feed forward of architecture')
	parser.add_argument('--data','-d',choices = DATASETS.keys(),default="mnist_binary")
	parser.add_argument('--L','-l',type=int,choices=[1,2],default=1)
	parser.add_argument('--alpha','-a',type=float,default=1.)

	args = parser.parse_args()
	data_name = args.data
	l = args.L
	alpha = args.alpha

	my_loader = Loader(data_name)
	data = my_loader.load()

	if l==1:
		model = net.VRalphaNet(data[0],[(200,'d'),(200,'d'),(50,'s')],'softplus')
	else:
		model = net.VRalphaNet(data[0],[(200,'d'),(200,'d'),(100,'s'),(100,'d'),(100,'d'),(50,'s')],'tanh')

	main(data,model,alpha)