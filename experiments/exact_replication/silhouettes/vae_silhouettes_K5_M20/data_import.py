from google.colab import drive
drive.mount('/content/drive')


### silhouettes
# # Load data with random initialized train/test split
if os.environ.get('CLOUDSDK_CONFIG') is not None:   
    fpath = "/content/drive/My Drive/data/caltech101_silhouettes_28.mat"
else:
    fpath = os.path.abspath('data/caltech101_silhouettes_28.mat')

data = loadmat(fpath) 
data = 1-data.get('X')

np.random.seed(seed)
np.random.shuffle(data)

num_train = int(.9* data.shape[0])

data_train = data[:num_train]
data_test = data[num_train:]

data_train_t, data_test_t = T(data_train), T(data_test)