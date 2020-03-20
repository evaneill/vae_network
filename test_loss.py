import argparse

from torch import Tensor as T

from models import network as net
from utils import Loader, DATASETS
from models.loss import VRBound

def main(data,model,alpha,optimize_on,batch_size):
	print("Turning into tensors...")
	data_train, data_test = data[0], data[1]
	data_train_t, data_test_t = T(data_train), T(data_test)

	print("Trying to feed forward...")
	output, qmu, qlog_sigma = model.forward(data_train_t[:batch_size,:])

	print(f"Trying to test loss with alpha={str(alpha)} and L={len(model.encoder.layers)}")
	print(f"==== Loss = {VRBound(alpha,model,output,qmu,qlog_sigma,optimize_on=optimize_on)} =====")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='test feed forward of architecture')
	parser.add_argument('--data','-d',choices = DATASETS.keys(),default="mnist_binary")
	parser.add_argument('--L','-l',type=int,choices=[1,2],default=1)
	parser.add_argument('--alpha','-a',type=float,default=1.)
	parser.add_argument('--optimize_on',choices=['full_lowerbound','max'],default='full_lowerbound')
	parser.add_argument('--n_samples','-k',type=int,default=5)
	parser.add_argument('--batch_size','-batch',type=int,default=100)

	args = parser.parse_args()
	data_name = args.data
	l = args.L
	alpha = args.alpha
	optimize_on=args.optimize_on
	n_samples = args.n_samples
	batch_size = args.batch_size

	my_loader = Loader(data_name)
	data = my_loader.load()

	if data_name=="freyfaces":
		data_type="continuous"
	else:
		data_type="binary"

	if l==1:
		model = net.VRalphaNet(data[0],[(200,'d'),(200,'d'),(50,'s')],'softplus',data_type=data_type,n_samples=n_samples)
	else:
		model = net.VRalphaNet(data[0],[(200,'d'),(200,'d'),(100,'s'),(100,'d'),(100,'d'),(50,'s')],'tanh',data_type=data_type)

	main(data,model,alpha,optimize_on,batch_size)