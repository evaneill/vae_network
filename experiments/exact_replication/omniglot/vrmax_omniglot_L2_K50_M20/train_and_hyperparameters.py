batch_size = 20 #@param {type:"slider", min:1,max:200}
test_batch_size = batch_size
# testing_frequency=100 # Depricated in favor of half of every round
# epochs = 1001
num_rounds=5
seed = 1
log_interval = 400
log_test_value = 100
K = 50 #@param {type:"slider", min:5, max:50, step:1}
learning_rate = 1e-4 # Was 5e-4, which overtrained
discrete_data = True
alpha = 0 #@param [0, 1] {type:"raw"}
cuda = torch.cuda.is_available()

data_name = 'omniglot' #@param['silhouettes','omniglot','freyfaces']

model_type = 'vrmax' #@param['iwae','vrmax','vae']
torch.manual_seed(seed)

logging_filename = f'{model_type}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)