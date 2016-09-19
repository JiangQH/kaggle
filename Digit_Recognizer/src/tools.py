import numpy as np
import os.path as osp

def load_data(data_dir):
    train_data = open(osp.join(data_dir, 'train.csv')).read()
    train_data = train_data.split("\n")[1:]
    train_data = [i.split(",") for i in train_data]
    print len(train_data)
    test_data = open(osp.join(data_dir, 'test.csv')).read()
    test_data = test_data.split("\n")[1:]
    print len(test_data)

    X_train = np.array([[int(i[j]) for j in range(1, len(i))] for i in train_data]) # each row is a pic
    y_train = np.array(int(i[0]) for i in train_data) # each row is a label
    print X_train.shape, y_train.shape

    X_test = np.array([[int(i[j]) for j in range(0, len(i))] for i in test_data]) # each row is a pic to predict
    print X_test.shape

    return X_train, y_train, X_test



load_data('../dataSet')