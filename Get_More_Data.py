import numpy as np
from decimal import *

class neural_Network():

    def NN_data(self):
        self.join_data = []
        join_data1 = []
        """Data Sets"""
        print('')
        X = np.array([[1], [2], [3], [4]])
        print('X = np.array([[1], [2], [3], [4]])')
        y = np.array([[5], [6], [7], [8]])
        print('y = np.array([[5], [6], [7], [8]])')
        """create the join_data 'ZERO' entry"""
        self.join_data.append(X[0, 0])
        self.join_data.append(y[0, 0])
        """Convert from list to array"""
        self.join_data = [self.join_data]
        """ADD the other pieces of data"""
        for i in range(1, 4):
            join_data1.append(X[i, 0])
            join_data1.append(y[i, 0])
            '''Convert from list to array'''
            join_data1 = [join_data1]
            self.join_data = np.concatenate([self.join_data, join_data1], axis=0)
            """reinitialize empty list"""
            join_data1 = []
    def getmoredata(self):
        self.NN_data()
        """Here we multiply each array element by some integer (1.1)"""
        for i in range(0, 4):
            more_data = np.array((self.join_data[[i]] * 1.1))
            self.join_data = np.concatenate([self.join_data, more_data], axis=0)
    def Reconstitute_DataSets(self):
        X =[]; y = []
        self.getmoredata()
        print('')
        for i in range(0, 8):
            index_0 = self.join_data[i, 0]
            index_1 = self.join_data[i, 1]
            X.append(index_0)
            X = [float(Decimal("%.2f" % e)) for e in X]
            y.append(index_1)
            y = [float(Decimal("%.2f" % e)) for e in y]
        X = [X]
        y = [y]
        print('New expanded X np.arrray data =')
        print(X)
        print('')
        print('New expanded y np.arrray data =')
        print(y)

NN = neural_Network()
#print(NN_data())
#print(NN. getmoredata())
print(NN.Reconstitute_DataSets())