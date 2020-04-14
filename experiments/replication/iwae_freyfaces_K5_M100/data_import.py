from google.colab import drive
drive.mount('/content/drive')

# Load data with random initialized train/test split
if os.environ.get('CLOUDSDK_CONFIG') is not None:   
    fpath = "/content/drive/My Drive/data/freyfaces.pkl"
else:
    fpath = os.path.abspath('data/freyfaces.pkl')

f = open(fpath,'rb')
data = pickle.load(f,encoding='latin1')
f.close()

np.random.seed(seed)
np.random.shuffle(data)

train_ratio = .9
num_train = int(train_ratio* data.shape[0])

data_train_t = T(1-data[:num_train])
data_test_t = T(1-data[num_train:])