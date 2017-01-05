from tools import SimpleTools as ST
from keras.engine.topology import Layer
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

class MyArrayIterator(NumpyArrayIterator):
    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg', flip_indices=None):
        super(MyArrayIterator, self).__init__(X, y,batch_size, shuffle, seed, dim_ordering,save_to_dir,
                                              save_prefix, save_format)
        self.flip_indices = flip_indices

    def next(self):
        batch_X, batch_y = super(MyArrayIterator, self).next()
        # flip them there
        batch_size = batch_X.shape[0]
        S = batch_X.shape[1]
        indices = np.random.choice(batch_size, batch_size/2, replace=False)
        batch_X[indices] = batch_X[indices, :, :, ::-1]

        if batch_y is not None:
            batch_y[indices, ::2] = S - batch_y[indices, ::2]
            for flip in self.flip_indices:
                batch_y[indices, flip[0]], batch_y[indices, flip[1]] = (
                    batch_y[indices, flip[1]], batch_y[indices, flip[0]]
                )
        return batch_X, batch_y




class FlippedImageDataGenerator(ImageDataGenerator):

    def flip_indices(self, flip=None):
        if flip is not None:
            self.flip = flip
        else:
            self.flip = [
                (0, 2), (1, 3),
                (4, 8), (5, 9), (6, 10), (7, 11),
                (12, 16), (13, 17), (14, 18), (15, 19),
                (22, 24), (23, 25),
                ]


    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        try:
            self.flip
        except NameError:
            self.flip_indices()

        return MyArrayIterator( X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, flip_indices=self.flip)













