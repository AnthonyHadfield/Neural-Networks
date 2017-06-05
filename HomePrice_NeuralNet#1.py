import numpy as np

#This code builds a nerual network around homes in my area.
#The home_feature are (beds, baths, Sqfeet, Lot_size).
#Neural network will have 4 inputs_layers (one for each feature data element), 4 hidden_layer_Nodes, and 1 Output_Node (4 x 1 vector)
#Data will be passed from each input_layer to each hidden_layer_Node, So each Hidden_layer_Notes have 4 weighted data_set involving 4 features.
#The Output_Layer_Node will be a 4 X 1 vector of weights.

home_features = np.array(([1, 1, 8750, 1132], [2, 2, 7800, 1382], [1, 1, 10000, 1825], [5, 3, 8100, 2340]), dtype=float)
home_price = np.array(([165000], [192500], [203000], [330000]),dtype=float)

#NORMALIZE data

home_features = home_features/np.amax(home_features)
home_price = (home_price)/max(home_price)

class Neural_Network(object):

    def __init__(self):
        self.inputLayerSize = 4
        self.hiddenLayerSize = 4
        self.outputLayerSize = 1
        self.Input_feature_weights = (np.random.rand(4, 4)) # sets up a 4 X 4 matrix of weights associated with the 4 (datasets) and 4 (features)
        self.Output_weights = (np.random.rand(4, 1)) # contains the 4 X 1 output weights

        print("Feature weights, ")
        print(self.Input_feature_weights)
        print('Output weights')
        print(self.Output_weights)

NN = Neural_Network()
print(NN)