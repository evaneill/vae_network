batch_size = 128
epochs = 101
seed = 1
log_interval = 1000
testing_frequency = 20
K = 5
learning_rate = 1e-3
discrete_data = True
alpha = 0
cuda = torch.cuda.is_available()
test_batch_size = 32
model_type = 'vrmax'
torch.manual_seed(seed)

data_name = 'mnist' 

logging_filename = f'{model_type}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)