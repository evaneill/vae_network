import config as cfg

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from .samplers import gaussianSampler 
from .layers import StochasticGaussianLayer, DeterministicLayer

# This implementation borrows a lot from the pytorch VAE example
class VRalphaNet(nn.Module):
	""" 'Renyi Divergence AutoEncoder Net' """

	def __init__(self,train_data,layers,activations,n_samples=5,alpha=1,data_type='binary'):
		"""Implements a fully connected FF network with stochastic and deterministic layers
		
		Args:
		    train_data (np.ndarray): array of data to reconstruct. Should be of the shape (N_obs, N_features). ONLY USED TO INFORM ARCHITECTURE
		
		    layers (list): list of tuples of the form (N neurons, "(s)tochastic") or (N neurons, "(d)terministic") (or for shorthand (N,"d"/"s"))
		    		(does not have to take into account input size)
		    		(all stochastic layers will have gaussian sampling)
		
		    activations (str): activation of all but final layer, whose activation is determined by the task and choice of loss. Must be exactly the name of an activation, i.e. "nn.<activations>" must be in the torch.nn library
		
		    n_samples (int, optional): Number of samples to generate for the decoder for each sample input at each stochastic layer (beware blowup with multiple stochastic layers!)
		
		    alpha (int, optional): Renyi divergence hyperparameter alpha
		    
		    data_type (str, optional): Either "binary" or "continuous" - determines the final activation of the decoder. sigmoid or gaussian (return both mu and sigma), respectively. 
		    							Also determines the reconstruction likelihood used: either bernoulli or gaussian		
		Raises:
		    BadInputException: Input is bad
		"""

		if layers[-1][1]!="s":
			print("The last layer you define in the layers argument has to be 's' bruh")
			raise BadInputException

		if False in [isinstance(k,tuple) for k in layers]:
			print("layers argument gotta be list of tuples bruh, see documentation")
			raise BadInputException

		super(VRalphaNet,self).__init__()

		self.encoder = EncoderNet(train_data,layers,activations,n_samples)
		self.decoder = DecoderNet(train_data,layers[::-1],activations,data_type)

	def forward(self,x):

		# Sampling is built into the logic of stochastic layers, which the encoder should end in
		q_samples, qmu, qlog_sigmasq = self.encoder.encode(x)

		output, pmu, plog_sigmasq = self.decoder.decode(q_samples[-1])

		return q_samples, qmu, pmu, qlog_sigmasq, plog_sigmasq


class EncoderNet(nn.Module):

	def __init__(self,data,layers,activations,n_samples):
		"""Summary
		
		Args:
		    data (numpy ndarray): Data input of size (N observations) x (N dimensions)
		    layers (list of ints): Size of dense hidden layers between 
		    activations (str): activation from torch.nn library that'll be used for all but stochastic layers
		    divergence_loss_alpha (float): alpha of divergence to be handed to divergence loss
		"""
		super(EncoderNet,self).__init__()

		augmented_layers = [(data.shape[1],'d')] + layers
		self.layers = []
		layer=[]
		for this_layer, next_layer in zip(augmented_layers[:-1],augmented_layers[1:]):
			if next_layer[1]=='d':
				layer.append(DeterministicLayer(this_layer[0],next_layer[0],activations)) 
			elif next_layer[1]=='s':
				layer.append(StochasticGaussianLayer(this_layer[0],next_layer[0]))
				self.layers.append(layer)
				layer=[]
			else:
				print(f"'{next_layer[1]}' isn't a valid type of layer - gotta be 's' or 'd' bruh")
				raise BadInputException

		self.n_samples = n_samples

	def encode(self,data):
		outputs = [data.repeat_interleave(self.n_samples,dim=0)]

		# Keep a list of these, since they'll have to be incorporated into the divergence measure
		# Length of these lists should be equal to the # of stochastic layers (minimum length 1)
		mu_list, log_sigmasq_list = [], []

		output = outputs[-1]
		for layer in self.layers:
			for unit in layer:
				output, mu, log_sigmasq = unit.forward(output)
				if mu is not None and log_sigmasq is not None:
					mu_list.append(mu)
					log_sigmasq_list.append(log_sigmasq)
					outputs.append(output)

		return outputs, mu_list, log_sigmasq_list
		


class DecoderNet(nn.Module):

	def __init__(self,data,layers,activations,data_type):
		"""Summary
		
		Args:
		    data (numpy ndarray): Data input of size (N observations) x (N dimensions)
		    layers (list of ints): Size of dense hidden layers between 
		    sampling (str): Either "(Bb)ernoulli" or "(Gg)aussian". This is the choice of latent parameters determined by encoder
		    activations (str): Either "tanh" or "softplus", as per paper's choices
		"""
		super(DecoderNet,self).__init__()

		augmented_layers = layers + [(data.shape[1],'d')]
		self.layers = []
		layer = []
		for this_layer, next_layer in zip(augmented_layers[:-2],augmented_layers[1:-1]):
			if next_layer[1]=='d':
				layer.append(DeterministicLayer(this_layer[0],next_layer[0],activations)) 
			elif next_layer[1]=='s':
				layer.append(StochasticGaussianLayer(this_layer[0],next_layer[0]))
				self.layers.append(layer)
				layer=[]
			else:
				print(f"'{next_layer[1]}' isn't a valid type of layer - gotta be 's' or 'd' bruh")
				raise BadInputException

		if data_type=='binary':
			layer.append(DeterministicLayer(augmented_layers[-2][0],augmented_layers[-1][0],'sigmoid'))
			self.layers.append(layer)
		elif data_type=='continuous':
			layer.append(StochasticGaussianLayer(augmented_layers[-2][0],augmented_layers[-1][0]))
			self.layers.append(layer)
		else:
			print(f"data_type {data_type} is invalid - should be 'binary' or 'continuous'")
			raise BadInputException

	def decode(self,data):

		mu_list, log_sigmasq_list = [], []
		outputs = [data]

		output = outputs[-1]
		for layer in self.layers:
			for unit in layer:
				output, mu, log_sigmasq = unit.forward(output)
				if mu is not None and log_sigmasq is not None:
					mu_list.append(mu)
					log_sigmasq_list.append(log_sigmasq)
					outputs.append(output)

		return outputs, mu_list, log_sigmasq_list

class BadInputException(Exception):
	pass
