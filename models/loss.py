
import torch
import torch.nn as nn
import torch.nn.functional as F

def renyiDivergence(self,alpha,q_mu,q_log_sigmasq,p_mu,p_log_sigmasq):
		"""
		It's expected that 
			(a) outputs are a tensor,  
			(b) each represents multiple samples over each image/sample, and
			(c) as per other implementations, use linear sigma activation to represent log of sigma^2
		"""
		if self.alpha==1:
			if len(q_mu)==1:
				return -.5 * torch.sum(1+q_log_sigmasq - q_mu.pow(2) - q_log_sigmasq.exp())
			else:
				output = 0

		elif self.alpha==0:
			pass
		else:
			pass

def gaussianLogLikelihoodLoss(output,data,n_samples):

	multiplied_data = torch.repeat_interleave(data,n_samples,dim=0)

	return

def bernoulliLogLikelihoodLoss(output,data,n_samples)
	
	multiplied_data = torch.repeat_interleave(data,n_samples,dim=0)

	return
