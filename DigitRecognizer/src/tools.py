import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    """
    :param train_path: the path to train file
    :param test_path: the path to test file
    :return: train_X, train_y, test_X
    """
    train = pd.read_csv('../dataset/train.csv')
    test = pd.read_csv('../dataset/test.csv')

    train_X = train.iloc[:, 1:].values
    train_X = np.float32(train_X) # each row is a pic
    train_y = train.iloc[:, 0].values
    train_y = np.int64(train_y) # each row is a label

    test_X = test.values
    test_X = np.float32(test_X)

    return train_X, train_y, test_X


def vis_data(data, label):
    """
    :param data:  the X feature data (m * n)
    :param label: the corresponding label (m * 1)
    :return: none
    """
    length = len(label)
    count = 8 # the number of pics to show each row
    for i in range(0, count):
        for j in range(0, count):
            idx = j * count + i + 1
            index = np.random.randint(0, length)
            l = label[index]
            img = data[index].reshape((28, 28))
            plt.subplot(count, count, idx)
            plt.imshow(np.uint8(img))
            plt.title(l)
            plt.axis("off")
            plt.subplots_adjust(wspace = 1.5)
    plt.show()







