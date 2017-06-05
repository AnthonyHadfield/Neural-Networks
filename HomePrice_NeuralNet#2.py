import numpy as np

# This code builds a nerual network around homes in my area.
# The home_feature are (beds, baths, Sqfeet, Lot_size).
# Neural network will have 4 inputs_layers (one for each feature data element), 4 hidden_layer_Nodes, and 1 Output_Node (4 x 1 vector)
# Data will be passed from each input_layer to each hidden_layer_Node, So each Hidden_layer_Notes have 4 weighted data_set involving 4 features.
# The Output_Layer_Node will be a 4 X 1 vector of weights.

home_features = np.array(([1, 1, 8750, 1132], [2, 2, 7800, 1382], [1, 1, 10000, 1825], [5, 3, 8100, 2340]), dtype=float)
home_price = np.array(([165000], [192500], [203000], [330000]), dtype=float)
X = home_features
y = home_price

#NORMALIZE data
X = X/np.amax(X)
y = y/max(y)

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 4
        self.hiddenLayerSize = 4
        self.outputLayerSize = 1

        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize) # sets up a 4 X 4 matrix of weights associated with 4 (homes) having 4 (features)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize) # contains the 4 X 1 output weights

    def forward(self, X):
        # Propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        print("z2 data")
        print(self.z2)
        self.a2 = self.sigmoid(self.z2)
        print("a2 data")
        print(self.a2)
        self.z3 = np.dot(self.a2, self.W2)
        print("z3 data")
        print(self.z3)
        yHat = self.sigmoid(self.z3)
        print("yHat")
        print(yHat)
        print("plus return yHat")
        return yHat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

NN = Neural_Network()

print(NN)
print(NN.forward(X))
