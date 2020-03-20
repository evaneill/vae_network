# Samplers should be functions that take mu, log_sigma tensors and return sampled versions
import torch

def gaussianSampler(mu, log_sigma):
	
	std = torch.exp(log_sigma) # exp(log(sigma))
	eps = torch.randn_like(std)

	return mu+eps*std

def bernoulliSampler(theta):
	# This is an undifferentiable operation, but I only plan on using the theta parameters so it's w/e

	return torch.bernoulli(theta)