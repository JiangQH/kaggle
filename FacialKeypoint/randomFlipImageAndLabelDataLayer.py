import caffe
import numpy as np
from random import shuffle
from tools import SimpleTools as ST
import os.path as osp


class RandomFlipImageAndLabelDataLayerSys(caffe.Layer):
    """
    Here define the sele implemented data layer
    to load the data and hold it. randomly flip the image and
    the corresponding y values in facial key point dectection
    """
    def setup(self, bottom, top):
        params = eval(self.param_str)
        # check it
        check_params(params)
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        # the batch loader
        self.batch_loader = BatchLoader(params)
        self.top_names = ['data', 'label']
        im, label = self.batch_loader.load_next_image_label()
        top[0].reshape(self.batch_size, 1, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, len(label))




    def forward(self, bottom, top):
        """
        the forward function
        :param bottom:
        :param top:
        :return:
        """
        for itt in range(self.batch_size):
            ims, label = self.batch_loader.load_next_image_label()
            top[0].data[itt, ...] = ims
            top[1].data[itt, ...] = label



    def reshape(self, bottom, top):
        """
        the reshape function
        :param bottom:
        :param top:
        :return:
        """
        pass


    def backward(self, top, propagate_down, bottom):
        """
        just pass
        :param top:
        :param propagate_down:
        :param bottom:
        :return:
        """
        pass





class BatchLoader(object):

    def __init__(self, params):
        self.path = params['path']
        self.im_shape = params['im_shape']
        self.selection = params['selection']
        self.random_flip = params['random_flip']
        self.cur = 0    # the current position
        self.transformer = ST()
        # load all the data in, since it will not occupy much memory
        self.X, self.y = self.transformer.loadHdf5Data(self.path)
        self.total_len = self.X.shape[0]
        self.sample_index = range(0, self.total_len)
        shuffle(self.sample_index)
        print 'BatchLoader loading {} images'.format(self.total_len)




    def load_next_image_label(self):
        """
        load the image and label, and decide whether to do the transform
        :return:
        """

        if self.cur == self.total_len:
            self.cur = 0
            print 'the whole dataset traveled, shuffing the dataset'
            shuffle(self.sample_index)

            # load img and the y label
        index = self.sample_index[self.cur]
        im = self.X[index, :]
        label = self.y[index, :]
        # do a random flip, if the random flip is set
        if self.random_flip:
            im, label = self.transformer.flipImage(im, label, self.selection)
            self.cur += 1
        return self.transformer.preprocess(im, label)






def check_params(params):
    """
    check the params for model params
    :param params:
    :return:
    """
    required = ['batch_size', 'path', 'im_shape', 'selection', 'random_flip']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
