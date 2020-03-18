import config as cfg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .samplers import BernoulliSampler, GaussianSampler


class VAENet(nn.Module):

	def __init__(self, data, layers, sampling, activations, n_samples=5):
		super(Net, self).__init__()

		self.encoder = EncoderNet(data, layers, sampling, activations)
		self.decoder = DecoderNet(layers[::-1], activations)

		self.n_samples = n_samples  # number of samples generated for decoder training

	def forward(self, x, n_samples):
		output_params = self.encoder.forward(x)

		samples = self.encoder.sampler(self.n_samples, output_params)

		output = self.decoder.forward(samples)

		return output


class EncoderNet(nn.Module):

	def __init__(self, data, layers, sampling, activation):
		"""Summary
		Args:
			data (numpy ndarray): Data input of size (N observations) x (N dimensions)
			layers (list of ints): Size of dense hidden layers between
			activation (str): Either "tanh" or "softplus", as per paper's choices
		"""
		# super(VAENet, self).__init__()

		self.in_to_hidden = nn.Linear(data.shape[1], 200)
		self.hidden_to_mean = nn.Linear(200, 20)
		self.hidden_to_var = nn.Linear(200, 20)

		self.activation = F.tanh if activation == "tanh" else F.softplus
		input_layer_size = data.shape[1]

	def forward(self, x):
		hidden = self.activation(self.in_to_hidden(x))
		mean = self.hidden_to_mean(hidden)
		var = self.hidden_to_var(hidden)
		eps = torch.randn_like(mean)
		# hopefully sqrt part works
		# alternatively: act as if network preditcs logvar and do
		# logvar = self.hidden_to_var(hidden)
		# z = mean + torch.exp(0.5*logvar)
		z = mean + torch.sqrt(var) * eps
		return z


class DecoderNet(nn.Module):

	def __init__(self, data, activation, data_continuous):
		"""Summary

		Args:
			data (numpy ndarray): Data input of size (N observations) x (N dimensions)
			layers (list of ints): Size of dense hidden layers between
			sampling (str): Either "(Bb)ernoulli" or "(Gg)aussian". This is the choice of latent parameters determined by encoder
			data_continous (bool): Either "(Bb)ernoulli" or "(Gg)aussian". This is the choice of latent parameters determined by encoder
			activations (str): Either "tanh" or "softplus", as per paper's choices
		"""
		self.lat_to_hidden = nn.Linear(20, 200)
		self.hidden_to_out = nn.Linear(200, data.shape[1])
		if data_continuous:
			self.hidden_to_var = nn.Linear(200, 20)

		self.sampler = BernoulliSampler() if data_continuous else BernoulliSampler()
		self.activation = F.tanh if activation == "tanh" else F.softplus
		self.data_continuous = data_continuous
		input_layer_size = data.shape[1]
	# super(Net,self).__init__()

	def forward(self, x):
		hidden = self.activation(self.lat_to_hidden(x))
		if self.data_continuous:
			return F.sigmoid(self.hidden_to_out(hidden))
		else:
			return self.hidden_to_out(hidden)

def loss(est, gt, mean, var, data_continuous):
	# todo think about how loss is actually computed xd

	# important: that loss is somewhat intractable for a != 1 might actually be relevant here
	# under stand what special procedure they used
	# im not sure if it is feasible to do just the importance weight backpropagation with pytorch -- might need to use
	# lower level libraries for that,,, maybe skip it then?
	# todo; change to appropriate one
	# taken from Pytorch reference paper
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + torch.log(var) - mean.pow(2) - var)
	if data_continuous:
		# todo: make sure this works -- ie look at how other implementations are doing this lmaokai
		# maybe i dont even need to sample here x_X
		# now that I think about it im pretty sure i dont
		noisy_estimate = torch.empty_like(est).normal_(mean=est)
		loss = torch.nn.MSELoss()
		likelihood_loss = loss(noisy_estimate, gt)
		return likelihood_loss + KLD
	else:
		# sum cause don't want to average but actually take the sum since that is equiv to Bernoulli
		return F.binary_cross_entropy(est, gt, reduction='sum')