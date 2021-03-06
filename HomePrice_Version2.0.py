from decimal import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
"""THIS is an artificial Neurul Net Version 1.0, with small dataset"""
"""Original DATA SET"""
home_features = np.array ([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11], [12], [13],[14], [15],[16],[17],
                           [18], [19], [20], [21], [22], [23], [24],[25], [26], [27], [28],[29], [30]])

home_price = np.array ([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11], [12], [13],[14], [15],[16],[17],
                           [18], [19], [20], [21], [22], [23], [24],[25], [26], [27], [28],[29], [30]])
hp_batch_1 = home_price
X = home_features
y = home_price
"""Here we normalize all data """
X = X / np.amax(X)
y = y / max(y)

"""y_ is the appended list of J values as we step through the traun method and decrease the Error """
y_ = []
"""X_ is the list of epoch numbers relative to y_ item."""
X_ = []
total = []

W1_update = []

class Home_Price(object):

    def tanh(self, z):
        return np.tanh(z)
    def tanhPrime(self, z):
        return 1.0 - np.tanh(z) ** 2

    def ReLU(self, z):
        return z * (z > 0)

    def ReLUPrime(self, z):
        return 1. * (z > 0)

    """A 12 train block was shown to be the most efficient"""
    def batches(self):

        self.hp_batch_1 = hp_batch_1

        self.hf_batch_1 = home_features
        self.hp_batch_1 = home_price
        """Normalize DATA"""

        self.hf_batch_1 = self.hf_batch_1 / np.amax(self.hf_batch_1)
        self.hp_batch_1 = self.hp_batch_1 / np.amax(self.hp_batch_1)

        """Get weights"""

        #self.batch1_W1 = np.random.normal(-0.7, 0.7, (1, 30))
        #self.batch1_W2 = np.random.normal(-0.7, 0.7, (30, 1))


        self.batch1_W1 = np.load('C:/Users/user/PycharmProjects/Neural_Nets/W1_update.npy')
        self.batch1_W2 = np.load('C:/Users/user/PycharmProjects/Neural_Nets/W2_update.npy')

    def train(self):

        self.batches()
        Epoch = 250001
        theta = 0.0514
        b = 0.25
        for i in range(2, Epoch):
            if i == 200:
                theta = 0.0514
                bias = 0.26
            if i == 250:
                theta = 0.03
                bias = 3
            if i == 100000:
                theta = 0.0245
                bias = 3

            z2 = (np.dot(self.hf_batch_1, self.batch1_W1) + b)
            a2 = self.tanh(z2)
            #a2 = self.ReLU(z2)
            z3 = (np.dot(a2, self.batch1_W2) + b)
            yHat = self.tanh(z3)
            #yHat = self.ReLU(z3)
            J = 0.5 * sum((self.hp_batch_1 - yHat) ** 2)
            y_.append(J)
            X_.append(i)
            """ Backpropagation"""
            delta3 = np.multiply(-(self.hp_batch_1 - yHat), self.tanhPrime(z3))
            #delta3 = np.multiply(-(self.hp_batch_1 - yHat), self.ReLUPrime(z3))
            dJdW2 = np.dot(a2.T, delta3)
            delta2 = np.dot(delta3, self.batch1_W2.T) * self.tanhPrime(z2)
            #delta2 = np.dot(delta3, self.batch1_W2.T) * self.ReLUPrime(z2)
            dJdW1 = np.dot(self.hf_batch_1.T, delta2)
            """Updating the weights: define next set of weights"""
            W1_grad_applied = np.dot(theta, dJdW1)
            W2_grad_applied = np.dot(theta, dJdW2)
            self.batch1_W1 = self.batch1_W1 - W1_grad_applied
            self.batch1_W2 = self.batch1_W2 - W2_grad_applied
            print('Epoch : {0}, Error = {1}'.format(i, J))


        for i in range(0, 30):
            #print('')
            accuracy = ((yHat[i] * 30) / hp_batch_1[i]) * 100
            print('%0.1f,        %0.1f,        %0.1f ' % ((hp_batch_1[i]), yHat[i] * 30, accuracy))




NN = Home_Price()

#print(NN.batches())
#print(NN.forward())
print(NN.train())


plt.scatter(X_[50000 :], y_[50000 :], s=25, c="blue")
plt.xlabel('Epochs # training cycles')
plt.ylabel('Error (Cost function J')
#plt.show()
