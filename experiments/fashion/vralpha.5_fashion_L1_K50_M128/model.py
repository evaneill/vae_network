class mnist1_model(nn.Module):
    def __init__(self):
        super(mnist1_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.K = K
    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, testing_mode=False,alpha=alpha):
        # data = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        mu, logstd = model.encode(data_k_vec)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)

        # summing over latents due to independence assumption
        # (B*K)
        log_q = compute_log_probabitility_gaussian(z, mu, logstd)

        log_p_z = compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False), torch.zeros_like(z, requires_grad=False))
        decoded = model.decode(z)
        if discrete_data:
            log_p = compute_log_probabitility_bernoulli(decoded, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p = compute_log_probabitility_gaussian(decoded, data_k_vec, torch.zeros_like(decoded))
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or testing_mode:
            log_w_matrix = (log_p_z + log_p - log_q).view(B, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p - log_q).view(B*K, 1)*1/K
        elif model_type=='general_alpha' or model_type=='vralpha':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1-alpha)
        elif model_type=='vrmax':
            log_w_matrix = (log_p_z + log_p - log_q).view(-1, K).max(axis=1,keepdim=True).values

        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        
        if model_type=='vralpha' and not testing_mode:
            sample_dist = Multinomial(1,ws_norm)
            ws_sum_per_datapoint = log_w_matrix.gather(1,sample_dist.sample().argmax(1,keepdim=True))
        else:
            ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        
        if model_type in ["general_alpha","vralpha"] and not testing_mode:
            ws_sum_per_datapoint/=(1-alpha)
        
        loss = -torch.sum(ws_sum_per_datapoint)

        return decoded,mu, logstd, loss