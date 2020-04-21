# Define likelihood functions
def compute_log_probabitility_gaussian(obs, mu, logstd, axis=1):
    # leaving out constant factor related to 2 pi in formula
    return torch.sum(-0.5 * ((obs-mu) / torch.exp(logstd)) ** 2 - logstd, axis)-.5*obs.shape[1]*T.log(torch.tensor(2*np.pi)) 

def compute_log_probabitility_bernoulli(theta, obs, axis=1):
    return torch.sum(obs*torch.log(theta+1e-18) + (1-obs)*torch.log(1-theta+1e-18), axis)