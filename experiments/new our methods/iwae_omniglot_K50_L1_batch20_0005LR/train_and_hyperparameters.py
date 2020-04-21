batch_size = 20
test_batch_size = 32
epochs = 501
seed = 1
log_interval = 100
log_test_value = 100
K = 50
learning_rate = 5e-4
discrete_data = True
alpha = 0
num_rounds = 6
cuda = torch.cuda.is_available()
logging_filename = f'iwae_omniglot_L1.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)

model_type = 'iwae'
torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")