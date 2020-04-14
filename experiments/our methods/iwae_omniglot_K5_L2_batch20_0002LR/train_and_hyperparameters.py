batch_size = 20
test_batch_size = 20
testing_frequency=50
epochs = 501
seed = 1
log_interval = 100
log_test_value = 100
K = 5
learning_rate = 2e-4
discrete_data = True
alpha = 0 #@param ["0", "1"] {type:"raw"}
cuda = torch.cuda.is_available()
logging_filename = f'iwae_omniglot_L2.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)

model_type = 'iwae' #@param['iwae','vrmax','vae']
torch.manual_seed(seed)