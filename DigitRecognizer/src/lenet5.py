from tools import DataHandler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# define the weight and bias init function
# the shape for weight: patchsize, input, output
def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# note in tf. the shape is [N H W C]. so for most strides the 0, and 3 entry should be 1
# strides = [1, stride, stride, 1]

def conv2d(x, W, b):
	conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	return tf.nn.bias_add(conv, b)

def fc(x, W, b):
	mul = tf.matmul(x, W)
	return tf.nn.bias_add(mul, b)


def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def relu(x):
	return tf.nn.relu(x)


def dense_to_one_hot(labels, labelcount):
	num_labels = labels.shape[0]
	index_offset = np.arange(num_labels) * labelcount
	labels_one_hot =  np.zeros((num_labels, labelcount))
	labels_one_hot.flat[index_offset + labels.ravel()] = 1
	return labels_one_hot


# define the model
def model(data, keep_prob):
	# data is the input here
	
	# the first conv layer with relu followed
	# this is a layer with (32 channels, 5*5 patch)
	conv1_weights = weight_variable([5, 5, 1, 32])
	conv1_bias = bias_variable([32])
	conv1 = conv2d(data, conv1_weights, conv1_bias)
	# followed the relu1
	relu1 = relu(conv1)
	print 'conv1 with shape ' + conv1.get_shape()

	# then the pooling layer with 2*2 and stride 2, this will downsample the layer
	pool1 = max_pool(relu1)

	# and a conv layer followed with (64 channels, 5*5 patch)
	conv2_weights = weight_variable([5, 5, 32, 64])
	conv2_bias = bias_variable([64])
	conv2 = conv2d(pool1, conv2_weights, conv2_bias)

	# followed by a relu layer
	relu2 = relu(conv2)
	print 'conv2 with shape ' + conv2.get_shape()

	# followed by a pooling layer
	pool2 = max_pool(relu2)
	print 'pool2 with shape ' + pool2.get_shape()

	# then flatten it to use a fully-connected layer, with 1024 output
	pool_shape = pool2.get_shape().as_list()
	pool2_flat = tf.reshape(pool2, pool_shape[0],
						pool_shape[1]*pool_shape[2]*pool_shape[3])
	fc1_weights = weight_variable([pool_shape[1]*pool_shape[2]*pool_shape[3],
								1024])
	fc1_bias = bias_variable([1024])
	fc1 = fc(pool2_flat, fc1_weights, fc1_bias)

	# then a relu layer
	relu_fc1 = relu(fc1)
	print 'fc1 with shape ' + relu_fc1.get_shape()
	# then a dropout layer
	relu_fc1 = dropout(relu_fc1, keep_prob)

	# then a final fc layer, with output equal to the label count
	fc2_weights = weight_variable([1024, 10])
	fc2_bias = bias_variable([10])
	fc2 = fc(relu_fc1, fc2_weights, fc2_bias)
	print 'fc2 with shape ' + fc2.get_shape()
	
	return fc2


def main(_):
	# get the data
	dh = DataHandler()
	train_X, train_y, test_X = dh.load_data('../dataset/train.csv', '../dataset/test.csv')
	# split the data into train and val
	X_train, y_train, X_val, y_val = train_test_split(train_X, train_y, random_state=42, test_size=0.1)
	img_size = X_train.shape[1]
	label_count = np.unique(y_train).shape[0]
	y_train = dense_to_one_hot(y_train, 10)
	y_val = dense_to_one_hot(y_val, 10)
	print 'training size {}'.format(X_train.shape)
	print 'val size {}'.format(X_val.shape)
	x = tf.placeholder('float', shape=[None, img_size])
	y_ = tf.placeholder('float', shape=[None, label_count])
	# load the model
	










