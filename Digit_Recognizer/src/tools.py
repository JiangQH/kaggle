import numpy as np
import pandas as pd

def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    y = train.label.values
    X = train.iloc[:, 1:].values
    test_x = test.values
    print('Load training data ' + str(len(y)) + ', testing data ' + str(len(test_x)))
    y = np.array(y)
    X = np.array(X)
    test_x = np.array(test_x)

    return y, X, test_x

