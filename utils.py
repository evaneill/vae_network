# Functions here depend on data being in the same form as referenced paper, including that some come in subdirectories!
import config as cfg

import sys, os

import pickle 
import numpy as np
from scipy.io import loadmat
from mnist import MNIST

DATASETS = {
	"freyfaces": "freyfaces.pkl",
	"mnist":{
		"train":"MNIST/",
		"test":"MNIST/"
	},
	"mnist_binary":{
		"train":"BinaryMNIST/binarized_mnist_train.amat",
		"test":"BinaryMNIST/binarized_mnist_test.amat",
		"valid":"BinaryMNIST/binarized_mnist_valid.amat"
	},
	"silhouettes":"caltech101_silhouettes_28.mat",
	"omniglot":"chardata.mat",
}

class Loader:

	def __init__(self,data_name):
		"""Instantiate the loader with functionality to load in one of the datasets from the paper
		
		Args:
		    data_name (str): One of "MNIST","MNIST_binary", "FreyFaces", "OMNIGLOT", "Silhouettes"
		"""
		if data_name.lower().strip() not in DATASETS.keys():
			print(f"{data_name} isn't a valid data name! One of "+", ".join(DATASETS.keys()))
			raise Exception

		self.data_name = data_name.lower().strip()

	def load(self,train_ratio=.9,seed=123):
		"""Load the data into a train and test np.ndarray
		
		Args:
		    train_ratio (float, optional): proportion of data to be used for training. Some datasets are already split and this is ignored
		
		Returns:
		    np.ndarray: (training data, test data)
		"""
		data_dir = cfg.DATA_DIR

		if isinstance(DATASETS.get(self.data_name),dict):
			
			if len(DATASETS.get(self.data_name))==2: # Relevant only for MNIST
				train_fpath = os.path.join(data_dir,DATASETS.get(self.data_name).get('train'))
				test_fpath = os.path.join(data_dir,DATASETS.get(self.data_name).get('test'))
			
			else: # Only relevant for binarized MNIST
				train_fpath = os.path.join(data_dir,DATASETS.get(self.data_name).get('train'))
				test_fpath = os.path.join(data_dir,DATASETS.get(self.data_name).get('test'))
				valid_fpath = os.path.join(data_dir,DATASETS.get(self.data_name).get('valid'))
		else:
			fpath = os.path.join(data_dir,DATASETS.get(self.data_name))

		print(f"Trying to load {self.data_name} from directory(ies):")
		
		if self.data_name == "freyfaces":
			# Load freyfaces
			print(f"...from {os.path.join(data_dir,fpath)}")
			f = open(fpath,'rb')
			data = pickle.load(f,encoding='latin1')
			f.close()

			# This block is directly from the VRBound repository
			np.random.seed(seed)
			np.random.shuffle(data)
			if train_ratio==1 or (0>train_ratio or 1<train_ratio):
				print(f"Train split ratio {train_ratio} or test value is invalid!")
				raise Exception
			num_train = int(train_ratio* data.shape[0])

			data_train = data[:num_train]
			data_test = data[num_train:]
			# End of copy

		elif self.data_name == "silhouettes":
			# Load silhouettes data
			print(f"...from {os.path.join(data_dir,fpath)}")

			# These lines are also from VRBound repository
			data = loadmat(fpath) 
			data = 1-data.get('X')

			# This block is directly from the VRBound repository
			np.random.seed(seed)
			np.random.shuffle(data)
			if train_ratio==1 or (0>train_ratio or 1<train_ratio):
				print(f"Train split ratio {train_ratio} or test value is invalid!")
				raise Exception
			num_train = int(train_ratio* data.shape[0])


			data_train = data[:num_train]
			data_test = data[num_train:]
			# End of copy

		elif self.data_name == "mnist":
			print("MNIST data is already train/test split - training ratio input ignored!")
			print(f"...from {os.path.join(data_dir,DATASETS.get(self.data_name)['train'])}")

			data_train, _ = MNIST(train_fpath).load_training() # We don't care about what the labels are; overwrite
			data_test, _ = MNIST(test_fpath).load_testing()

		elif self.data_name == "mnist_binary":
			print("MNIST data is already train/test split - training ratio input ignored!")
			print(f"...from {os.path.join(train_fpath.split('/')[-2])}")
			# This is directly from the iwae codebase
			def lines_to_np_array(lines):
			    return np.array([[int(i) for i in line.split()] for line in lines])
			with open(train_fpath) as f:
			    lines = f.readlines()
			train_data = lines_to_np_array(lines).astype('float32')
			with open(test_fpath) as f:
			    lines = f.readlines()
			validation_data = lines_to_np_array(lines).astype('float32')
			with open(valid_fpath) as f:
			    lines = f.readlines()
			data_test = lines_to_np_array(lines).astype('float32')

			data_train= np.concatenate([train_data, validation_data], axis=0)

		elif self.data_name == "omniglot":
			print(f"...from {os.path.join(data_dir,fpath)}")
			print("Omniglot data is already train/test split - training ratio input ignored!")

			data = loadmat(fpath)

			# From iwae repository
			data_train = data['data'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F') 
			data_test = data['testdata'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
		
		return data_train, data_test