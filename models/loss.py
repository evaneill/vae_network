
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch import Tensor as T
import torch

def renyiDivergence(alpha,model,q_samples):
		"""
		Args:
		    alpha (float): alpha of renyi alpha-divergence
		    model (VRalphaNet): net from models.network
		    q_samples (list): list of the output latent samples form training, with the data as the first element.
		    	(should be the result of model.forward(data))
		
		Returns:
		    float: Renyi alpha-divergence of the latent model parameters
		"""
		alpha = float(alpha)
		if abs(alpha-1)<=1.-1e-5:
			# log_likelihood_running_sum = -.5 * torch.sum(1+q_log_sigmasq[-1] - q_mu[-1].pow(2) - q_log_sigmasq[-1].exp())
			log_likelihood_running_sum = log_likelihood_samples_to_gaussian(q_samples[-1],torch.zeros_like(q_samples[-1]),torch.ones_like(q_samples[-1]))
			if len(q_samples)==1:
				return log_likelihood_running_sum
			else:
				for q_layer, p_layer, sample, next_sample in zip(model.encoder.layers,model.decoder.layers[::-1],q_samples,q_samples[1:]):
					q_out, p_out = sample, next_sample
					for q_unit,p_unit in zip(q_layer,p_layer): # Rely on them being the same size
						q_out, q_mu, q_sigma = q_unit.forward(q_out)
						p_out, p_mu, p_sigma = p_unit.forward(p_out)

					log_likelihood_running_sum+=log_likelihood_samples_to_gaussian(sample,p_mu,p_sigma) -log_likelihood_samples_to_gaussian(next_sample,q_mu,q_sigma)

				return log_likelihood_running_sum
		else:
			# assume alpha !=0 and alpha!=1
			# Calculate weights

			# Calculate estimate

			pass


def gaussianLogLikelihoodLoss(output,data,n_samples):

	multiplied_data = torch.repeat_interleave(data,n_samples,dim=0)

	return

def bernoulliLogLikelihoodLoss(output,data,n_samples):
	
	multiplied_data = torch.repeat_interleave(data,n_samples,dim=0)

	return

def log_likelihood_samples_to_gaussian(samples,mean,sigma):
	# really similar to IWAE code. expecting break.
	output =  -.5*T.log(torch.tensor(2*np.pi))*samples.shape[1]
	return output - .5*T.sum(T.pow((samples-mean)/sigma,2)+2*T.log(sigma),axis=1)