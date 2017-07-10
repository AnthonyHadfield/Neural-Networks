import numpy as np

def join_data():

    join_data = []
    join_data1 = []

    """Data Sets"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([[5], [6], [7], [8]])

    """create the join_data 'ZERO' entry"""
    join_data.append(X[0, 0])
    join_data.append(y[0, 0])
    """Convert from list to array"""
    join_data = [join_data]

    """ADD the other pieces of data"""
    for i in range(1, 4):

        join_data1.append(X[i, 0])
        join_data1.append(y[i, 0])

        '''Convert from list to array'''
        join_data1 = [join_data1]

        join_data = np.concatenate([join_data, join_data1], axis=0)
        """reinitialize empty list"""
        join_data1 = []

    print('')
    print('Join_data np.array = ')
    print('')
    print(join_data)

print(join_data())