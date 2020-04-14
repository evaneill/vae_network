batch_size = 20
test_batch_size = 32
epochs = 501
seed = 1
log_interval = 100
log_test_value = 100
K = 5
learning_rate = 5e-4
discrete_data = True
num_rounds = 6
cuda = torch.cuda.is_available()

alpha = .5
model_type = 'vralpha'

torch.manual_seed(seed)

data_name = 'omniglot'

device = torch.device("cuda" if cuda else "cpu")

if model_type!="general_alpha":
	model_name=model_type
else:
	model_name = model_type+str(alpha)

logging_filename = f'{model_name}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)