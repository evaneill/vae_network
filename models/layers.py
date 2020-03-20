# Stochastic and Deterministic layers (both fully connected)
from .samplers import gaussianSampler, bernoulliSampler

import torch.nn as nn
from torch.nn import functional as F

import math

from torch import Tensor as T
import numpy as np
import torch

class Linear(nn.Linear):

	def reset_parameters(self):
		bound = .1 * math.sqrt(6/(self.in_features+self.out_features)) # from iwae codebase
		nn.init.uniform_(self.weight,a=-bound,b=bound)
		if self.bias is not None:
			nn.init.zeros_(self.bias) #lol lazy cheat


class StochasticGaussianLayer(nn.Module):

	def __init__(self,prev_layer_neurons,n_neurons):
		# N_neurons = number used to represent both mean and SD of stochastic layer
		# No activation needed - stochastic layers apparently ALWAYS use linear for mu and exp(linear) for sigma^2
		super(StochasticGaussianLayer,self).__init__()

		self.fc_mu = Linear(prev_layer_neurons,n_neurons)
		self.fc_log_sigma = Linear(prev_layer_neurons,n_neurons)

		self._type = "gaussian"

	def forward(self,input_activations):
		mu = self.fc_mu(input_activations)
		log_sigma = self.fc_log_sigma(input_activations)

		return gaussianSampler(mu,log_sigma), mu, log_sigma

	def _params(self):
		return torch.nn.ParameterList(self.fc_mu,self.fc_log_sigma)

class StochasticBernoulliLayer(nn.Module):

	def __init__(self,prev_layer_neurons,n_neurons):
		# N_neurons = number used to represent both mean and SD of stochastic layer
		super(StochasticBernoulliLayer,self).__init__()

		self.fc_theta = Linear(prev_layer_neurons,n_neurons)
		
		self._type = "bernoulli"

	def forward(self,input_activations):
		theta = torch.sigmoid(self.fc_theta(input_activations))

		# Nondifferentiable but don't need it to be - only output theta matters for our purpose ¯\_(° ¿ °)_/¯
		return bernoulliSampler(theta), theta, None

	def _params(self):
		return self.fc_theta

class DeterministicLayer(nn.Module):

	def __init__(self,prev_layer_neurons,n_neurons,activation):
		
		super(DeterministicLayer,self).__init__()

		self.fc = Linear(prev_layer_neurons,n_neurons)

		try:
			self.activation = getattr(torch,activation)
		except:
			self.activation = getattr(F,activation)

	def forward(self,input_activations):
		return self.activation(self.fc(input_activations)), None, None

	def _params(self):
		return self.fc




