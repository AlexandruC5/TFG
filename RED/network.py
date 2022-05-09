#Init network

import random
from random import seed



#Init de la red on se li fique una hidden layer y un output layer que sera el retorn. 
def init_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer =[{'weights': [random() for i in range(n_inputs +1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden +1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network



seed(1)
network = init_network(2,1,2)

for layer in network: print(layer)
