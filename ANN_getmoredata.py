from decimal import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
"""Original DATA SET"""
"""NOTE the home_price is 1/10 to avoid some long number issues"""
"""HOME FEATURE is Square feet"""
home_features = np.array ([[2250], [3811], [3000], [1724], [1325], [2550], [1484], [1072], [2013], [4430], [3583], [4850]])
home_price = np.array([[80000], [128000], [107000], [50600], [28500], [92660], [41400], [16600], [66300],
                       [135100], [119400], [141200]])

plt.scatter(home_features, home_price, s=50, c="red")
plt.title('ORIGINAL DATA SET')
plt.show()

more_home_features = []
more_home_prices = []

X = home_features
y = home_price

class Get_More_Data(object):

    """Put home_features and home_price into ONE data ARRAY"""
    def Join_Data(self):

        self.join_data = []
        join_data1 = []

        """create the join_data[0] 'ZERO' entry"""
        self.join_data.append(home_features[0, 0])
        self.join_data.append(home_price[0, 0])

        """Convert from list to array"""
        self.join_data = [self.join_data]

        """ADD the other pieces of data"""
        for i in range(1, 12):
            join_data1.append(home_features[i, 0])
            join_data1.append(home_price[i, 0])
            '''Convert from list to array'''
            join_data1 = [join_data1]
            self.join_data = np.concatenate([self.join_data, join_data1], axis=0)
            """reinitialize empty list"""
            join_data1 = []
        print('')

    def getmoredata(self):

        self.Join_Data()
        """INCREASE DATA by 10 %"""
        for i in range(0, 12):
            moredata = np.array((self.join_data[[i]] * 1.1))
            self.join_data = np.concatenate([self.join_data, moredata], axis=0)

    def Reconstitute_DataSets(self):

        self.getmoredata()

        X = []
        y = []

        for i in range(0, 24):

            index_0 = self.join_data[i, 0]
            index_1 = self.join_data[i, 1]
            X.append(index_0)
            """Here we normalize some long decimal point issues"""
            X = [float(Decimal("%.2f" % e)) for e in X]
            y.append(index_1)
            y = [float(Decimal("%.2f" % e)) for e in y]

        """RECONSTITUE ARRAYS"""
        X = np.array([X])
        y = np.array([y])

        """HERE We reset Home Price AS To Orininal by * 10"""
        y = y *10

        more_home_features = X
        more_home_prices = y

        plt.scatter(more_home_features, more_home_prices, s=50, c="blue")
        plt.title('GET MORE DATA SET')
        plt.show()

DATA = Get_More_Data()
print(DATA.Reconstitute_DataSets())