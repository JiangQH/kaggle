import pandas
import numpy as np
import matplotlib.pyplot as plt

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

    def load_data(self, path, phase='Train', select=None):
        """
        the load data phase. transform data to 96 * 96
        :param path:
        :param phase:
        :param select:
        :return:
        """
        df = pandas.read_csv(path)
        print df.count()
        if select is not None:
            df = df[list(self.dict[select]) + ['Image']]
        df = df.dropna()  # drop the nan
        X_s = df['Image'].values
        X = [np.fromstring(iterm, sep=' ').reshape((96, 96)) for iterm in X_s]
        df = df[df.columns[:-1]] # remove the Image column
        if phase == 'Train':
            y = df.values
            return np.asarray(X), np.asarray(y)
        return np.asarray(X)

    def plot(self, X, y, axis):
        """
        plot the img and the y point in img
        :param X:
        :param y:
        :return:
        """
        axis.scatter(y[0::2], y[1::2], marker='x', s=10)
        axis.imshow(X, cmap='gray')

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
        # flip the y, the y_axis should be swapped while the x_axis should be flipped
        y_s = y.copy()
        y_s[0::2] = X_s.shape[0] - y_s[0::2]
        if select is None:
            select = 'all'
        flips = self.flip_dict[select]
        for flip in flips:
            temp = y_s[flip[0]]
            y_s[flip[0]] = y_s[flip[1]]
            y_s[flip[1]] = temp
        return X_s, y_s



train_X, train_y = SimpleTools().load_data('./data/training.csv')
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
X, y = train_X[0], train_y[0]
SimpleTools().plot(X, y, ax)

X_f, y_f = SimpleTools().flipImage(X, y)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
SimpleTools().plot(X_f, y_f, ax)
test = SimpleTools().load_data('./data/test.csv')
