from utils import Loader

import models.network as network

my_loader = Loader("MNIST_binary")
data_train, data_test = my_loader.load()

my_net = network.RDAENet(data_train,[(200,'d'),(200,'d'),(50,'s')],'tanh')