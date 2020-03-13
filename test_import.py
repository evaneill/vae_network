from data_loader import Loader

import argparse

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Test data imports')
	parser.add_argument("--data","-D",choices=["MNIST","MNIST_binary","freyfaces","OMNIGLOT","silhouettes"])

	args = parser.parse_args()
	data_name = args.data

	loader = Loader(data_name)
	print(f"test {data_name}")

	train, test = loader.load()

	print("Done :)")
