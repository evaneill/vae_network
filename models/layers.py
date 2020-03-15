# Stochastic and Deterministic layers (both fully connected)
from .samplers import gaussianSampler

import torch.nn as nn
from torch.nn import functional as F
import torch

class StochasticGaussianLayer:

	def __init__(self,prev_layer_neurons,n_neurons):
		# N_neurons = number used to represent both mean and SD of stochastic layer
		# No activation needed - stochastic layers apparently ALWAYS use linear for mu and exp(linear) for sigma^2
		self.fc_mu = nn.Linear(prev_layer_neurons,n_neurons)
		self.fc_log_sigmasq = nn.Linear(prev_layer_neurons,n_neurons)

	def forward(self,input_activations):
		mu = self.fc_mu(input_activations)
		log_sigmasq = self.fc_log_sigmasq(input_activations)

		return gaussianSampler(mu,log_sigmasq), mu, log_sigmasq

class DeterministicLayer:

	def __init__(self,prev_layer_neurons,n_neurons,activation):
		
		self.fc = nn.Linear(prev_layer_neurons,n_neurons)

		try:
			self.activation = getattr(torch,activation)
		except:
			self.activation = getattr(F,activation)

	def forward(self,input_activations):
		return self.activation(self.fc(input_activations)), None, None




