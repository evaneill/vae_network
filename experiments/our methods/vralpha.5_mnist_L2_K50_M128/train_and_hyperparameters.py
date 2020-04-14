batch_size = 128
epochs = 101
seed = 1
log_interval = 1000
testing_frequency = 20
K = 50
learning_rate = 1e-3
discrete_data = True
cuda = torch.cuda.is_available()
test_batch_size = 32
torch.manual_seed(seed)

alpha = .5
model_type = 'vralpha'

data_name = 'mnist' 

if model_type in ['vralpha','general_alpha']:
	model_name = model_type+str(alpha)
else:
	model_name = model_type

logging_filename = f'{model_name}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)