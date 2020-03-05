import config as cfg

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from .samplers import BernoulliSampler, GaussianSampler

class VAENet(nn.Module):

	def __init__(self,data,layers,sampling,activations,n_samples=5):
		super(Net,self).__init__()

		self.encoder = EncoderNet(data,layers,sampling,activations)
		self.decoder = DecoderNet(layers[::-1],activations)

		self.n_samples = n_samples # number of samples generated for decoder training

	def forward(self,x,n_samples):
		output_params = self.encoder.forward(x)

		samples = self.encoder.sampler(self.n_samples,output_params)

		output = self.decoder.forward(samples)

		return output


class EncoderNet(nn.Module):

	def __init__(self,data,layers,sampling,activations):
		"""Summary
		
		Args:
		    data (numpy ndarray): Data input of size (N observations) x (N dimensions)
		    layers (list of ints): Size of dense hidden layers between 
		    sampling (str): Either "(Bb)ernoulli" or "(Gg)aussian". This is the choice of latent parameters determined by encoder
		    activations (str): Either "tanh" or "softplus", as per paper's choices
		"""
		super(Net,self).__init__()
 
		if sampling.lower()=="bernoulli":
			self.sampler = BernoulliSampler()
		elif sampling.lower()=="gaussian":
			self.sampler == GaussianSampler()
		else:
			print(f"{sampling} is an invalid input!")
			exit()
		
		input_layer_size = data.shape[1]



class DecoderNet(nn.Module):

	def __init__(self,layers,activation):
		"""Summary
		
		Args:
		    data (numpy ndarray): Data input of size (N observations) x (N dimensions)
		    layers (list of ints): Size of dense hidden layers between 
		    sampling (str): Either "(Bb)ernoulli" or "(Gg)aussian". This is the choice of latent parameters determined by encoder
		    activations (str): Either "tanh" or "softplus", as per paper's choices
		"""
		super(Net,self).__init__()
		
		input_layer_size = data.shape[1]
