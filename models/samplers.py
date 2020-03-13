# Samplers should be functions that take mu, log_sigmasq tensors and return sampled versions
import torch

def gaussianSampler(mu, log_sigmasq):
	
	std = torch.exp(.5*log_sigmasq)
	eps = torch.randn_like(std)

	return mu+eps*std