import numpy as np
"""
# This code builds a nerual network around homes in my area.
# The home_feature are (beds, baths, Sqfeet, Lot_size).
# Neural network will have 4 inputs_layers (one for each feature data element), 4 hidden_layer_Nodes, and 1 Output_Node (4 x 1 vector)
# Data will be passed from each input_layer to each hidden_layer_Node, So each Hidden_layer_Notes have 4 weighted data_set involving 4 features.
# The Output_Layer_Node will be a 4 X 1 vector of weights.
Next we genarate the data arrays, and normalize the data.
"""
home_features = np.array(([1, 1, 8750, 1132], [2, 2, 7800, 1382], [1, 1, 10000, 1825], [5, 3, 8100, 2340]), dtype=float)
home_price = np.array(([165000], [192500], [203000], [330000]), dtype=float)
X = home_features
y = home_price
#NORMALIZE data
X = X/np.amax(X)
y = y/max(y)
"""
We not define a class function for our neutal net called Home_Price, we define the and initialize the network
and generate a weight matrix for the home_features, and a weight vector for the home_price data.
"""
class Home_Price(object):
    def __init__(self):
        self.inputLayerSize = 4
        self.hiddenLayerSize = 4
        self.outputLayerSize = 1
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize) # sets up a 4 X 4 matrix of weights associated with 4 (homes) having 4 (features)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize) # contains the 4 X 1 output weights
        """ Next in 'def fordward' we process our data and weight matrices (W1) through the hidden layer (sigmoid function), apply a second set
            of weights (W2), which all gets put through the output neuron (sigmoid function) to give a 1st pass estimate of
            a home_price yHat."""
    def forward(self, X):
        # Propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        print("y = House_Price")
        print(y*330000)
        yHat = (yHat*330000).astype(int)
        print("yHat = House_Price estimates")
        print(yHat)
        return yHat
        """Next we define the sigmoid function, and it's derivative sigmoidPrime"""
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    """Input home_price are (y), and the resultant estimate of home prices are yHat.
        In order to improve our estimate we develop what is called the cost function or Error
        between y and yHat given in 'def cost function'"""
    def costFunction(self, X, y):
        #compute cost for given X, y, use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        print("Error (J) is = ")
        return J
    """
      J is called the Cost or the Error, and in order to minimize this function we need to differentiate across J.
      This is done by a process called Stochastic Gradient Descent otherwise know as Back Propogation.
      This process is performed in stages and at each stage we make a small adjustments to the weights,
      in the (-) negative direction to the gradient, which is the result of differentiating over f(J).
      So J = 0.5*sum(y-yHat) squared. First we differentiate the whole function J = f(y)squared, which 
      gives us (y-yHat), next we differentiate the inner function (y-yHat) by taking the partial derivatives
      of its components. Diff y = 0, so we are left with differentiating yHat = f(f(z2)z3).
      To find d yHat/d z3 we differentiate the sigmoid function to get 'sigmoidPrime.'
      """
    def sigmoidPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    """This leads us into the next step of defining the BackPropogation algorithm which we get by taking
    the partial derivatives of yHat = f(f(z2)z3). Which we will get to in #5"""
NN = Home_Price()
print(NN.costFunction(X, y))

