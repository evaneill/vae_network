batch_size = 20
test_batch_size = 20
testing_frequency=50
epochs = 501
seed = 1
log_interval = 100
log_test_value = 100
K = 50
learning_rate = 2e-4
discrete_data = True
cuda = torch.cuda.is_available()

torch.manual_seed(seed)

data_name = 'omniglot'

alpha = .5
model_type = 'vralpha'

device = torch.device("cuda" if cuda else "cpu")

if model_type!="general_alpha" and model_type!="vralpha":
	model_name=model_type
else:
	model_name = model_type+str(alpha)

logging_filename = f'{model_name}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)

