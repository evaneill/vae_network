# Samplers should be functions that take mu, log_sigmasq tensors and return sampled versions
import torch

def gaussianSampler(mu, log_sigmasq):
	
	std = torch.exp(.5*log_sigmasq) # exp(.5 log(o^2))
	eps = torch.randn_like(std)

	return mu+eps*std

def bernoulliSampler(theta):
	# This is an undifferentiable operation, but I only plan on using the theta parameters so it's w/e

	return torch.bernoulli(theta)