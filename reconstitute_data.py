import numpy as np
from decimal import *
data = np.array([[1, 5], [2, 6], [3, 7], [4, 8], [1.1, 5.5], [2.2, 6.6], [3.3, 7.7], [4.4, 8.8]], dtype=float)
def Reconstitute_DataSets():
    X = [];
    y = []
    print('')
    for i in range(0, 8):
        index_0 = data[i, 0]
        index_1 = data[i, 1]
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
print(Reconstitute_DataSets())