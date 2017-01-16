import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py
import os.path as osp
import os
from random import shuffle

class SimpleTools(object):
    def __init__(self):
        self.dict = {'eye_center': ('left_eye_center_x', 'left_eye_center_y',
                             'right_eye_center_x', 'right_eye_center_y'),
                     'nose': ('nose_tip_x', 'nose_tip_y'),
                     'mouth': ('mouth_left_corner_x', 'mouth_left_corner_y',
                                'mouth_right_corner_x', 'mouth_right_corner_y',
                                'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
                               ),
                     'mouth_bottom': ('mouth_center_bottom_lip_x','mouth_center_bottom_lip_y'),
                     'eye_inner': ('left_eye_inner_corner_x', 'left_eye_inner_corner_y',
                                    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                                    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                                    'right_eye_outer_corner_x', 'right_eye_outer_corner_y'),
                     'eye_outer': ('left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                                    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
                                    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                                    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y')
                     }
        self.flip_dict = {
            'eye_center': ((0, 2), (1, 3)),
            'nose': (),
            'mouth': ((0, 2), (1, 3)),
            'mouth_bottom': (),
            'eye_inner': ((0, 2), (1, 3), (4, 6), (5, 7)),
            'eye_outer': ((0, 2), (1, 3), (4, 6), (5, 7)),
            'all': ((0, 2), (1, 3),(4, 8), (5, 9), (6, 10), (7, 11),
                    (12, 16), (13, 17), (14, 18), (15, 19),(22, 24), (23, 25))
        }

        self.scale = 255
        self.constant = 48
        self.lookUpTable = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
                            'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
                            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
                            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
                            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x',
                            'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y',
                            'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
                            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
                            'mouth_center_bottom_lip_y']


    def load_data(self, path, phase='Train', select=None):
        """
        the load data phase. transform data to 96 * 96
        :param path:
        :param phase:
        :param select:
        :return:
        """
        df = pandas.read_csv(path)
        #print df.count()
        if select is not None:
            df = df[list(self.dict[select]) + ['Image']]
        df = df.dropna()  # drop the nan
        X_s = df['Image'].values
        X = [np.fromstring(iterm, sep=' ').reshape((96, 96)) for iterm in X_s]
        df = df[df.columns[:-1]] # remove the Image column
        if phase != 'Test':
            y = df.values
            return np.asarray(X), np.asarray(y)
        return np.asarray(X)


    def plot(self, X, y, save_name=None):
        """
        plot the img and the y point in img
        :param X:
        :param y:
        :return:
        """
        plt.figure()
        plt.imshow(X, cmap='gray')
        plt.scatter(y[0::2], y[1::2], marker='x', s=10)
        plt.show()
        if save_name is not None:
            plt.savefig(save_name)


    def flipImage(self, X, y, select=None):
        """
        flip the img and the corresponding label
        :param X:
        :param y:
        :param select:
        :return:
        """
        # flip the X
        X_s = X[:, ::-1]
        y_s = y.copy()
        y_s[0::2] = X_s.shape[0] - y_s[0::2]
        if select is None:
            select = 'all'
        flips = self.flip_dict[select]
        for flip in flips:
            temp = y_s[flip[0]]
            y_s[flip[0]] = y_s[flip[1]]
            y_s[flip[1]] = temp
        # flip the y, the y_axis should be swapped while the x_axis should be flipped
        return X_s, y_s



    def preprocess(self, X, y=None):
        X /= self.scale
        if y is not None:
            y = (y - self.constant) / self.constant
            return X, y
        return X

    def getTruey(self, y):
        y = y * self.constant + self.constant
        y[y>96] = 96
        y[y<0] = 0
        return y

    def umcompress(self, X, y=None):
        X *= self.scale
        if y is not None:
            y = y * self.constant + self.constant
            return X, y
        return X

    def getOutNum(self, split='all'):
        outs = {'all': 30, 'eye_center': 4,
                    'nose': 2, 'mouth': 6,
                    'mouth_bottom': 2,
                    'eye_inner': 8,
                    'eye_outer': 8}
        return outs[split]


    def splitTrainingData(self, path, select=None):
        """
        split the training data into train and val
        :param select:
        :return:
        """
        df = pandas.read_csv(path)
        if select is None:
            select = 'all'
        if select != 'all':
            df = df[list(self.dict[select]) + ['Image']]
        df.dropna(inplace=True)
        X_s = df['Image'].values
        X = [np.fromstring(iterm, sep=' ').reshape((96, 96)) for iterm in X_s]
        df = df[df.columns[:-1]]
        y = df.values
        label_names = list(df.columns.values)
        X = np.asarray(X)
        y = np.asarray(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # save to disk
        # write to file with the name holds
        save_path = osp.dirname(osp.abspath(path))
        train = h5py.File(osp.join(save_path, select + '_train.hdf5'), 'w')
        train.create_dataset('data', data=X_train, dtype='f8')
        train.create_dataset('label', data=y_train, dtype='f8')
        train.create_dataset('attr', data=label_names, dtype='S10')
        train.close()

        val = h5py.File(osp.join(save_path, select+'_val.hdf5'), 'w')
        val.create_dataset('data', data=X_val, dtype='f8')
        val.create_dataset('label', data=y_val, dtype='f8')
        val.create_dataset('attr', data=label_names, dtype='S10')
        val.close()

    def loadHdf5Data(self, path):
        f = h5py.File(path, 'r')
        #y = f['label'][:]
        return f['data'][:], f['label'][:]


    def splitTrainingDataToFile(self, path, select=None):
        df = pandas.read_csv(path)
        if select is None:
            select = 'all'
        if select != 'all':
            df = df[list(self.dict[select]) + ['Image']]
        df.dropna(inplace=True)
        X_s = df['Image'].values
        X = [np.fromstring(iterm, sep=' ').reshape((96, 96)) for iterm in X_s]
        df = df[df.columns[:-1]]
        y = df.values
        label_names = list(df.columns.values)
        X = np.asarray(X)
        y = np.asarray(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # write to file with the name holds
        save_path = osp.dirname(osp.abspath(path))
        train_path = osp.join(save_path, 'train_'+select)
        val_path = osp.join(save_path, 'val_'+select)
        if not osp.exists(train_path):
            os.mkdir(train_path)
        if not osp.exists(val_path):
            os.mkdir(val_path)
        train_list = []
        for it in range(X_train.shape[0]):
            save_name_x = osp.join(train_path, str(it)+'x.txt')
            save_name_y = osp.join(train_path, str(it)+'y.txt')
            train_list.append(save_name_x + ' ' + save_name_y)
            x = X_train[it, :, :]
            y = y_train[it, :]
            np.savetxt(save_name_x, x)
            np.savetxt(save_name_y, y)
        val_list = []
        for it in range(X_val.shape[0]):
            save_name_x = osp.join(val_path, str(it) + 'x.txt')
            save_name_y = osp.join(val_path, str(it) + 'y.txt')
            val_list.append(save_name_x + ' ' + save_name_y)
            x = X_val[it, :, :]
            y = y_val[it, :]
            np.savetxt(save_name_x, x)
            np.savetxt(save_name_y, y)

        shuffle(train_list)
        shuffle(val_list)
        with open(osp.join(save_path, 'train.txt'), 'w') as f:
            f.write("\n".join(train_list))
            f.close()
        with open(osp.join(save_path, 'val.txt'), 'w') as f:
            f.write("\n".join((val_list)))
            f.close()


    def loadImgAndLabel(self, path):
        paths = path.split(' ')
        img_path = paths[0]
        label_path = paths[1]
        img = np.loadtxt(img_path)
        label = np.loadtxt(label_path)
        return img, label

    def readLists(self, path):
        with open(path, 'rU') as f:
            lists = f.read().split('\n')
        return lists

    def getIndex(self, feature_names):
        index = []
        for name in feature_names:
            index.append(self.lookUpTable.index(name))
        return index

    def write_prediction(self, prediciton, rowid, outpath):
        dataframe = pandas.DataFrame({'RowId':rowid, 'Location': prediciton})
        dataframe.to_csv(outpath, index=False, header=True)

    def getFeatureName(self, model_name):
        if model_name != 'all':
            return list(self.dict[model_name])
        else:
            return self.lookUpTable











