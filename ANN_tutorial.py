import numpy as np
from  scipy import optimize
import matplotlib.pyplot as plt

X = np.array(([3, 5], [5, 1], [10, 1]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

#Normalize

X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

class Neural_Network(object):
    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self):

        #Propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        print('yHat', self.yHat)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Now we can compute the difference between out ist pass result yHat and the know House_Price (y)
    # def costFunction gives this result.

    def sigmoidPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self):
        #compute cost for given X, y, use weights already stored in class
        self.forward()
        J = 0.5*sum((y-self.yHat)**2)
        print('')
        print('J =', J)

    """
    J is called the Cost or the Error, and in order to minimize this function we need to differentiate across J.
    This is done by a process called Stochastic Gradient Descent otherwise know as Back Propogation.
    This process is performed in stages and at each stage we make a small adjustments to the weights,
    in the (-) negative direction to the gradient, which is the result of differentiating over f(J).
    So J = 0.5*sum(y-yHat) squared. First we differentiate the whole function J = f(y)squared, which 
    gives us (y-yHat), next we differentiate the inner function (y-yHat) by taking the partial derivatives
    of its components. Diff y = 0, so we are left with differentiating yHat = f(f(z2)z3).
    To find d yHat/d z3    
    """

    def costFunctionPrime(self):
        #compute the derivative with respect to W1 and W2
        self.forward()

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        print(self.W1)
        print(self.W2)
        print('')

        print("dJdW1 and dJdW1 gradients")
        print(dJdW1)
        print(dJdW2)

        print('')

        print(self.W1 + dJdW1)

        print('')

        print(self.W2 + dJdW2)

    #Helper functions for intereacting with other method/classes
    def getParams(self):
        #Get W1 and W2 Rolled into a vector:

        params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],
                (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)

        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradients(NN, X, y):
        paramsInitial = NN.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #set pertubation vector
            perturb[p] = e
            NN.setParams(paramsInitial + perturb)
            loss2 = NN.costFunction(X, y)

            NN.setParams(paramsInitial - perturb)
            loss1 = NN.costFunction(X, y)

            #Compute Numerical Gradient:
            numgrad[p] = (loss2-loss1) / (2*e)

            #Return the value we changed back to zero:
            perturb[p] = 0

        NN.setParams(paramsInitial)

        return numgrad

from scipy import optimize

class trainer(object):
    def __init__(self, NN):
        # Make Local reference to network:
        self.NN = NN

    def callbackF(self, params):
        self.NN.setParams(params)
        self.J.append(self.NN.costFunction(self.X, self.y))

        print(self.NN.costFunction(self.X, self.y))
        print(self.J)

    def costFunctionWrapper(self, params, X, y):
        self.NN.setParams(params)
        cost = self.NN.costFunction(X, y)
        grad = self.NN.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback

        self.X = X
        self.y = y

        self.J = []

        params0 = self.NN.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0,  jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackF)

        self.NN.setParams(_res.x)
        self.optimizationResults = _res

data = Neural_Network()
# data.forward()
# data.costFunction()
data.costFunctionPrime()