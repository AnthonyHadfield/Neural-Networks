import numpy as np

def getmoredata():

    X_y = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])

    print('')

    for i in range(0, 4):
        more_data = np.array((X_y[[i]] * 1.1))
        X_y = np.concatenate([X_y, more_data], axis=0)

    print(X_y)

print(getmoredata())