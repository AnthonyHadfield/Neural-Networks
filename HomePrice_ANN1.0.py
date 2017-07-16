from decimal import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

"""THIS is an artificial Neurul Net Version 1.0, with small dataset"""
"""Original DATA SET"""
home_features = np.array ([[1072],[1325],[1484],[1724], [2013],[2250],[2550],[3000], [3583],[3811], [4430], [4850]])
home_price = np.array([[16600],[28500],[41400],[50600], [66300],[80000],[92660],[107000],[119400],[128000],[135100], [141200] ])
home_prices = np.array([[166000],[285000],[414000],[506000], [663000],[800000],[926600],[1070000],[1194000],[1280000],[1351000], [1412000] ])

plt.scatter(home_features, home_prices, s=50, c = "red")
plt.title('ZILLOW PRICE')
plt.show()

X = home_features
y = home_price
"""Normalize DATA"""
X = X / np.amax(X)
y = y / max(y)

"""y_ is the appended list of J values as we step through the train method and decrease the Error """
y_ = []
"""X_ is the list of epoch numbers relative to y_ item."""
X_ = []
total = []

class Home_Price(object):

    def __init__(self):

        self.inputLayerSize = 1
        self.hiddenLayerSize = 12
        self.outputLayerSize = 1

        self.W1 = np.random.normal(0, 1, (1, 12))
        self.W2 = np.random.normal(0, 1, (12, 1))

    def forward(self, X):

        total = 0

        self.z2 = (np.dot(X, self.W1))
        self.a2 = self.sigmoid(self.z2)
        self.z3 = (np.dot(self.a2, self.W2))
        self.yHat = self.sigmoid(self.z3)
        self.J = 0.5 * sum((y - self.yHat) ** 2)

        return self.J

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    """BackPropogation"""
    def costFunctionPrime(self, X, y):
        # compute the derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        self.dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        self.dJdW1 = np.dot(X.T, delta2)
        # return self.dJdW2

    """Train the ANN"""
    def train(self):

        self.costFunctionPrime(X, y)

        Epoch = 5000;
        theta = 0.9

        self.W1_update = self.W1
        self.W2_update = self.W2
        self.dJdW1_update = self.dJdW1
        self.dJdW2_update = self.dJdW2

        W1_grad_applied = np.dot(theta, self.dJdW1_update)
        W2_grad_applied = np.dot(theta, self.dJdW2_update)
        self.W1_update = self.W1_update - W1_grad_applied
        self.W2_update = self.W2_update - W2_grad_applied

        for i in range(2, Epoch):

            """Forward propagation"""
            self.z2 = np.dot(X, self.W1_update)
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2, self.W2_update)
            self.yHat = self.sigmoid(self.z3)
            J = 0.5 * sum((y - self.yHat) ** 2)

            y_.append(J)
            X_.append(i)

            """ Backpropagation"""
            delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
            self.dJdW2_update = np.dot(self.a2.T, delta3)
            delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
            self.dJdW1_update = np.dot(X.T, delta2)

            """Updating the weights: define next set of weights"""
            W1_grad_applied = np.dot(theta, self.dJdW1_update)
            W2_grad_applied = np.dot(theta, self.dJdW2_update)
            self.W1_update = self.W1_update - W1_grad_applied
            self.W2_update = self.W2_update - W2_grad_applied

            print('Epoch : {0}, Error(cost function) = {1}'.format(i, J))

            """Saving trained weights"""
            np.save('C:/Users/user/PycharmProjects/Neural_Nets/W1_update', self.W1_update)
            np.save('C:/Users/user/PycharmProjects/Neural_Nets/W2_update', self.W2_update)

        print('')
        for i in range(0, 12):
            result = self.yHat[i] * 1412000
            accuracy = ((self.yHat[i] * 1412000) / home_price[i]) * 100
            print('%0.1f,        %0.1f,        %0.1f ' % ((home_price[i]*10), self.yHat[i] * 1412000,  accuracy/10))

        print('HOME PRICE      ANN Estimate   % Accuracy')
        print('')
        print('NOTE THIS IS ON A SMALL DATASET SO LOW ACCURACY')

    def test_prep(self):

        """Train Data"""
        home_features = np.array([[1072], [1325], [1484], [1724], [2013], [2250], [2550], [3000], [3583], [3811], [4430], [4850]])
        home_price = np.array([[16600], [28500], [41400], [50600], [66300], [80000], [92660], [107000], [119400], [128000], [135100],
             [141200]])

        X = home_features
        y = home_price

        print('')
        """Test data"""
        test_features = np.array([[1880]])
        test_price = np.array([[32775]])

        """Combined test data SET"""
        testX_data = np.concatenate([X, test_features], axis=0)
        testy_data = np.concatenate([y, test_price], axis=0)

        """Here we normalize all data """
        testX_data = testX_data / np.amax(testX_data)
        testy_data = testy_data / max(testy_data)

        """Now add the test weights to the update weights"""
        self.W1_update = np.load('C:/Users/user/PycharmProjects/Neural_Nets/W1_update.npy')
        self.W2_update = np.load('C:/Users/user/PycharmProjects/Neural_Nets/W2_update.npy')

        self.W1_test = np.random.normal(0, 1, (1, 1))
        self.W2_test = np.random.normal(0, 1, (1, 1))

        """Combine test weight data SET"""
        testX_W1 = np.concatenate([self.W1_update.T, self.W1_test.T], axis=0)
        testX_W2 = np.concatenate([self.W2_update, self.W2_test], axis=0)

        return testX_data, testy_data, testX_W1, testX_W2, test_price

    def test(self):

        testX_data, testy_data, testX_W1, testX_W2, test_price = self.test_prep()

        self.z2 = (np.dot(testX_data, testX_W1.T))
        self.a2 = self.sigmoid(self.z2)
        self.z3 = (np.dot(self.a2, testX_W2))
        self.yHat = self.sigmoid(self.z3)

        print('Test Price #1')
        print(test_price)

        '''Test Estimate'''
        # print(test_price)

        #print(self.yHat)
        print('Price estimate')
        print(self.yHat[12] * 1412000)

        """We return self.yHat so that it can be used by later methods"""
        # return self.J


NN = Home_Price()

#print(NN.forward(X))

print(NN.train())

# print(NN.costFunctionPrime(X, y))

#print(NN.test_prep())

print(NN.test())

#print(NN.Join_Data())

#print(NN.getmoredata())

#print(NN.Reconstitute_DataSets())

plt.scatter(X_, y_, s=25, c="blue")
plt.xlabel('Epochs # training cycles')
plt.ylabel('Error (Cost function J')

#plt.scatter(home_features, home_price, s=50, c = "red")
#plt.ylabel('Zillow Price')
plt.show()

print('')