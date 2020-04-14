batch_size = 100 #@param {type:"slider", min:1,max:200}
test_batch_size = batch_size
# testing_frequency=100 # Depricated in favor of half of every round
epochs = 2001
seed = 1
log_interval = 10
log_test_value = 100
K = 5 #@param {type:"slider", min:5, max:50, step:1}
learning_rate = 5e-4 
discrete_data = True
alpha = 0 #@param [0, 1] {type:"raw"}
cuda = torch.cuda.is_available()

data_name = 'freyfaces' #@param['silhouettes','omniglot','freyfaces']

model_type = 'iwae' #@param['iwae','vrmax','vae']
torch.manual_seed(seed)

# Whether to store some gradients for study during training runtime
collect_grads=False

logging_filename = f'{model_type}_{data_name}_K{K}_M{batch_size}.log'
logging.basicConfig(filename=logging_filename,level=logging.DEBUG)