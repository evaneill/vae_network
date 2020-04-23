class mnist2_model(nn.Module):
    # data_dim, q_dims, latents_dim, p_dims
    def __init__(self):
        super(mnist2_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 100)
        self.fc32 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc61 = nn.Linear(100, 50)
        self.fc62 = nn.Linear(100, 50)

        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc91 = nn.Linear(100, 100)
        self.fc92 = nn.Linear(100, 100)

        self.fc10 = nn.Linear(100, 200)
        self.fc11 = nn.Linear(200, 200)
        self.fc12 = nn.Linear(200, 784)

        self.K = K

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mu1, logstd1 = self.fc31(h2), self.fc32(h2)
        z1 = self.reparameterize(mu1, logstd1)

        h3 = torch.tanh(self.fc4(z1))
        h4 = torch.tanh(self.fc5(h3))
        mu2, logstd2 = self.fc61(h4), self.fc62(h4)
        z2 = self.reparameterize(mu2, logstd2)
        return z1, mu1, logstd1, z2, mu2, logstd2

    def decode_for_testing(self, z2):
        # h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z3))
        h4 = torch.tanh(self.fc11(h3))
        return torch.sigmoid(self.fc12(h4))

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z1, z2):
        #h3 = F.relu(self.fc3(z))
        h1 = torch.tanh(self.fc7(z2))
        h2 = torch.tanh(self.fc8(h1))
        mu3, logstd3 = self.fc91(h2), self.fc92(h2)

        #z3 = self.reparameterize(mu3, logstd3)

        h3 = torch.tanh(self.fc10(z1))
        h4 = torch.tanh(self.fc11(h3))
        return z1, mu3, logstd3, torch.sigmoid(self.fc12(h4))

        # # h3 = F.relu(self.fc3(z))
        # h1 = torch.tanh(self.fc7(z2))
        # h2 = torch.tanh(self.fc8(h1))
        # mu3, logstd3 = self.fc91(h2), self.fc92(h2)
        # z3 = self.reparameterize(mu3, logstd3)
        #
        # h3 = torch.tanh(self.fc10(z3))
        # h4 = torch.tanh(self.fc11(h3))
        # return z3, mu3, logstd3, torch.sigmoid(self.fc12(h4))

    def forward(self, x):
        #mu, logstd = self.encode(x.view(-1, 784))
        z1, _, _, z2, mu, logstd = self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logstd)
        _, _, _, recon = self.decode(z1, z2)
        return recon, mu, logstd

    def compute_loss_for_batch(self, data, model, K=K, testing_mode=False):
        # data = (B, F) = (B, 1, H, W)
        B, _, H, W = data.shape
        data_k_vec = data.repeat((1, K, 1, 1)).view(-1, H*W)
        z1, mu1, logstd1, z2, mu2, logstd2 = model.encode(data_k_vec)
        # (B*K, #latents)
        log_q_1 = compute_log_probabitility_gaussian(z1, mu1, logstd1)
        log_q_2 = compute_log_probabitility_gaussian(z2, mu2, logstd2)

        log_p_z = compute_log_probabitility_gaussian(z2, torch.zeros_like(z2, requires_grad=False), torch.zeros_like(z2, requires_grad=False))
        #log_p_z = torch.sum(-0.5 * z2 ** 2, 1)
        z3, mu3, logstd3, recon = model.decode(z1, z2)
        log_p_1 = compute_log_probabitility_gaussian(z3, mu3, logstd3)
        if discrete_data:
            log_p_2 = compute_log_probabitility_bernoulli(recon, data_k_vec)
        else:
            # Gaussian where sigma = 0, not letting sigma be predicted atm
            log_p_2 = compute_log_probabitility_gaussian(recon, data_k_vec, torch.zeros_like(recon))
            # log_p = torch.sum(-0.5 * (decoded-data_k.view(-1, 784))**2, 1)
        # hopefully this reshape operation magically works like always
        if model_type == 'iwae' or testing_mode:
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B, K)
        elif model_type =='vae':
            # treat each sample for a given data point as you would treat all samples in the minibatch
            # 1/K value because loss values seemed off otherwise
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(B*K, 1)*1/K
            return 0, 0, 0, -torch.sum(log_w_matrix)
        elif model_type == 'vrmax':
            log_w_matrix = (log_p_z + log_p_1 + log_p_2 - log_q_1 - log_q_2).view(-1, K).max(axis=1,keepdim=True).values
            return 0, 0, 0, -torch.sum(log_w_matrix)
        log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
        ws_matrix = torch.exp(log_w_minus_max)
        ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
        # ws_norm = ws_matrix / torch.clamp(torch.sum(ws_matrix, 1, keepdim=True), 1e-9, np.inf)

        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
        loss = -torch.sum(ws_sum_per_datapoint)

        return recon, 0, 0, loss