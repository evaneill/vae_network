from google.colab import drive
drive.mount('/content/drive')

# Load data with random initialized train/test split
if os.environ.get('CLOUDSDK_CONFIG') is not None:   
    fpath = "/content/drive/My Drive/data/chardata.mat"
else:
    fpath = os.path.abspath('data/chardata.mat')

data = loadmat(fpath)

# From iwae repository
data_train = data['data'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F') 
data_test = data['testdata'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')

data_train_t, data_test_t = T(data_train), T(data_test)