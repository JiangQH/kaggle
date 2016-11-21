import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    """
    Simple data handler, used to load, write and display the data
    """
    def __init__(self):
        pass

    def load_data(self, train_path, test_path):
        """
        :param train_path: the path to train file
        :param test_path: the path to test file
        :return: train_X, train_y, test_X
        """
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        train_X = train.iloc[:, 1:].values
        train_X = np.float32(train_X) # each row is a pic
        train_y = train.iloc[:, 0].values
        train_y = np.int64(train_y) # each row is a label

        test_X = test.values
        test_X = np.float32(test_X)

        return train_X, train_y, test_X

    def write_data(self, prediction, outputname):
        dataframe = pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": np.uint8(prediction)})
        dataframe.to_csv(outputname, index=False, header=True)


    def vis_data(self, data, label):
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
                plt.imshow(np.uint8(img), cmap=cm.binary)
                plt.title(l)
                plt.axis("off")
                plt.subplots_adjust(wspace = 1.5)
        plt.show()

    def preprocess(self, data, scale=255.0):
	    data = np.multiply(data, 1.0 / scale)
	    return data

    def dense_to_one_hot(self, labels, labelcount):
	    num_labels = labels.shape[0]
	    index_offset = np.arange(num_labels) * labelcount
	    labels_one_hot =  np.zeros((num_labels, labelcount))
	    labels_one_hot.flat[index_offset + labels.ravel()] = 1
	    return labels_one_hot







