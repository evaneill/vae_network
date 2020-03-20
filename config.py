import os

# This workaround is to be able to look at data in your google drive on colab rather than uploading data to a colab directory every time
if os.environ.get('CLOUDSDK_CONFIG') is not None:	
	DATA_DIR = "/content/drive/My Drive/data/"
else:
	DATA_DIR = os.path.abspath('data/')

