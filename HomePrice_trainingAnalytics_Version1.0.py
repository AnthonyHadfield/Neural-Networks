from decimal import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

"""TRAINING suffers from variance in the weights.WHAT WE WANT is to define the best set of weights.
THIS CODE DOES THAT, it lets you run a selected number of Epoch's, and ADD up all of the J's.
This is repeated with new weight data in  for j in range(1, 6):
 We can now identify THE weight set that give THE lowest Average_total_J.
 These weights CAN now be ported into the regular ANN train method."""

home_features = np.array ([[1072],[1200],[1325],[1400],[1484],[1600], [1724], [1850], [1953],[2100], [2250],[2450],[2600], [2750], [3000],
                           [3150], [3300], [3450], [3583],[3811], [4000], [4330], [4600], [4850]])
home_price = np.array([[16600],[22800],[28500],[34000],[41000],[46000], [50600], [58000], [66000],[74000],[80000],[86000], [92660],[98500],
                       [107000],[110000], [115000],[119500], [122400],[128000],[132000], [135100],[138000], [141200] ])

"""Batch size of 12 was best"""
#hp_batch_1 = home_price[: -12]

class Home_Price(object):
    def tanh(self, z):
        return np.tanh(z)
    def tanhPrime(self, z):
        return 1.0 - np.tanh(z) ** 2
    def batches(self):
        self.hf_batch_1 = home_features[: -12]
        self.hp_batch_1 = home_price [: -12 ]
        """Normalize DATA"""
        self.hf_batch_1 = self.hf_batch_1 / np.amax(self.hf_batch_1)
        self.hp_batch_1 = self.hp_batch_1 / np.max(self.hp_batch_1)

    """HERE you try different theta's"""
    def best_theta(self):
        self.batches()
        Epoch = 20000
        theta = 0.08 #BEST
        count = 0
        b = 1
        """Here set # of data set you want to see"""
        for k in range(1, 6):
            Average_J = []
            for j in range(1, 6):
                """Here you can try different ranges"""
                self.batch1_W1 = np.random.normal(-1, 1, (1, 12))
                self.batch1_W2 = np.random.normal(-1, 1, (12, 1))

                total_J = 0
                Ave_total_J = 0

                """Here SET your Epoch size"""
                for i in range(2, Epoch):
                    z2 = (np.dot(self.hf_batch_1, self.batch1_W1) + b)
                    a2 = self.tanh(z2)
                    z3 = (np.dot(a2, self.batch1_W2) + b)
                    yHat = self.tanh(z3)
                    J = 0.5 * sum((self.hp_batch_1 - yHat) ** 2)
                    """ Backpropagation"""
                    delta3 = np.multiply(-(self.hp_batch_1 - yHat), self.tanhPrime(z3))
                    dJdW2 = np.dot(a2.T, delta3)
                    delta2 = np.dot(delta3, self.batch1_W2.T) * self.tanhPrime(z2)
                    dJdW1 = np.dot(self.hf_batch_1.T, delta2)
                    """Updating the weights: define next set of weights"""
                    W1_grad_applied = np.dot(theta, dJdW1)
                    W2_grad_applied = np.dot(theta, dJdW2)
                    self.batch1_W1 = self.batch1_W1 - W1_grad_applied
                    self.batch1_W2 = self.batch1_W2 - W2_grad_applied

                    total_J = total_J  +J
                    Ave_total_J = total_J/i

                Average_J.append(Ave_total_J)

            print(max(Average_J), min(Average_J))


NN = Home_Price()

print(NN.best_theta())