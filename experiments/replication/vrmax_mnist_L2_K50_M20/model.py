# Define the model
class mnist2_model(nn.Module):
    def __init__(self):
        super(mnist2_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 100) # stochastic 1
        self.fc32 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100,100)
        self.fc5 = nn.Linear(100,100)
        self.fc61 = nn.Linear(100,50) # Innermost (stochastic 2)
        self.fc62 = nn.Linear(100,50)

        self.fc7 = nn.Linear(50,100)
        self.fc8 = nn.Linear(100,100)
        self.fc81 = nn.Linear(100,100) # stochastic 1
        self.fc82 = nn.Linear(100,100)

        self.fc9 = nn.Linear(100, 200)
        self.fc10 = nn.Linear(200, 200)
        self.fc11 = nn.Linear(200, 784) # reconstruction

        self.K = K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu, log_std = self.fc31(h2), self.fc32(h2)

        z1 = self.reparameterize(mu, log_std)
        h3 = torch.tanh(self.fc4(z1))
        h4 = torch.tanh(self.fc5(h3))

        return self.fc61(h4), self.fc62(h4), [x,z1]

    def reparameterize(self, mu, logstd,test=False):
        std = torch.exp(logstd)
        if test==True:
          eps = torch.zeros_like(mu)
        else:
          eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z,test=False):
        #h3 = F.relu(self.fc3(z))
        h5 = torch.tanh(self.fc7(z))
        h6 = torch.tanh(self.fc8(h5))
        mu, log_std = self.fc81(h6), self.fc82(h6)

        z1 = self.reparameterize(mu, log_std,test=test)
        h7 = torch.tanh(self.fc9(z1))
        h8 = torch.tanh(self.fc10(h7))

        return torch.sigmoid(self.fc11(h8))

    def forward(self, x):
        mu, logstd, _= self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K,test=False):
        # data = (N,560)
        if model_type=='vae':
            alpha=1
        elif model_type in ('iwae','vrmax'):
            alpha=0
        else:
            # use whatever alpha is defined in hyperparameters
            if abs(alpha-1)<=1e-3:
                alpha=1

        data_k_vec = data.repeat_interleave(K,0)

        mu, log_std , [x,z1] = self.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, log_std)

        # Log p(z) (prior)
        log_p_z = torch.sum(-0.5 * z ** 2, 1)-.5*z.shape[1]*T.log(torch.tensor(2*np.pi))

        # q (z | h1)
        log_qz_h1 = compute_log_probabitility_gaussian(z, mu, log_std)

        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu, log_std = self.fc31(h2), self.fc32(h2)

        # q (h1 | x)
        log_qh1_x = compute_log_probabitility_gaussian(z1, mu, log_std)

        h5 = torch.tanh(self.fc7(z))
        h6 = torch.tanh(self.fc8(h5))
        mu, log_std = self.fc81(h6), self.fc82(h6)

        # log p(h1 | z)
        log_ph1_z = compute_log_probabitility_gaussian(z1,mu,log_std)

        h7 = torch.tanh(self.fc9(z1))
        h8 = torch.tanh(self.fc10(h7))

        decoded = torch.sigmoid(self.fc11(h8))

        # log p(x | h1)
        log_px_h1 = compute_log_probabitility_bernoulli(decoded,x)
        
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or test==True:
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, 1)*1/K
            return decoded,mu, log_std, -torch.sum(log_w_matrix)
        elif model_type=='general_alpha':
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_ph1_z + log_px_h1 - log_qz_h1 - log_qh1_x).view(-1, K).max(axis=1,keepdim=True).values
            return decoded, mu, log_std,-torch.sum(log_w_matrix)

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded, mu, log_std, loss