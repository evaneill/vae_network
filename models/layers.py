# Stochastic and Deterministic layers (both fully connected)
from .samplers import gaussianSampler, bernoulliSampler

import torch.nn as nn
from torch.nn import functional as F


from torch import Tensor as T
import numpy as np
import torch

class StochasticGaussianLayer:

	def __init__(self,prev_layer_neurons,n_neurons):
		# N_neurons = number used to represent both mean and SD of stochastic layer
		# No activation needed - stochastic layers apparently ALWAYS use linear for mu and exp(linear) for sigma^2
		self.fc_mu = nn.Linear(prev_layer_neurons,n_neurons)
		self.fc_log_sigmasq = nn.Linear(prev_layer_neurons,n_neurons)

		self._type = "gaussian"

	def forward(self,input_activations):
		mu = self.fc_mu(input_activations)
		log_sigmasq = self.fc_log_sigmasq(input_activations)

		return gaussianSampler(mu,log_sigmasq), mu, log_sigmasq

class StochasticBernoulliLayer:

	def __init__(self,prev_layer_neurons,n_neurons):
		# N_neurons = number used to represent both mean and SD of stochastic layer
		self.fc_theta = nn.Linear(prev_layer_neurons,n_neurons)
		
		self._type = "bernoulli"

	def forward(self,input_activations):
		theta = torch.sigmoid(self.fc_theta(input_activations))

		# Nondifferentiable but don't need it to be - only output theta matters for our purpose ¯\_(° ¿ °)_/¯
		return bernoulliSampler(theta), theta, None

class DeterministicLayer:

	def __init__(self,prev_layer_neurons,n_neurons,activation):
		
		self.fc = nn.Linear(prev_layer_neurons,n_neurons)

		try:
			self.activation = getattr(torch,activation)
		except:
			self.activation = getattr(F,activation)

	def forward(self,input_activations):
		return self.activation(self.fc(input_activations)), None, None




