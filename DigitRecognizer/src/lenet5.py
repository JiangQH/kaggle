from tools import DataHandler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
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




# define the model
def model(data, keep_prob):
    # the model is
    # data --> conv1(5,5,32) --> relu1 --> pool1(2,2) --> conv2(5, 5, 64)
    # --> relu2 --> pool2(2,2) --> flaten --> fc1(4096) --> relu_fc1 --> dropout_fc1
    # --> fc2(10) --> softmax(prediciton)
    # data is the input here

    # the first conv layer with relu followed
    # this is a layer with (32 channels, 5*5 patch)
    conv1_weights = weight_variable([5, 5, 1, 32])
    conv1_bias = bias_variable([32])
    conv1 = conv2d(data, conv1_weights, conv1_bias)
    # followed the relu1
    relu1 = relu(conv1)
    print 'conv1 with shape {}'.format(conv1.get_shape())

    # then the pooling layer with 2*2 and stride 2, this will downsample the layer
    pool1 = max_pool(relu1)

    # and a conv layer followed with (64 channels, 5*5 patch)
    conv2_weights = weight_variable([5, 5, 32, 64])
    conv2_bias = bias_variable([64])
    conv2 = conv2d(pool1, conv2_weights, conv2_bias)

    # followed by a relu layer
    relu2 = relu(conv2)
    print 'conv2 with shape {}'.format(conv2.get_shape())

    # followed by a pooling layer
    pool2 = max_pool(relu2)
    print 'pool2 with shape {}'.format(pool2.get_shape())

    # then flatten it to use a fully-connected layer, with 1024 output
    pool_shape = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [-1,
                        pool_shape[1]*pool_shape[2]*pool_shape[3]])
    fc1_weights = weight_variable([pool_shape[1]*pool_shape[2]*pool_shape[3],
                                1024])
    fc1_bias = bias_variable([1024])
    fc1 = fc(pool2_flat, fc1_weights, fc1_bias)

    # then a relu layer
    relu_fc1 = relu(fc1)
    print 'fc1 with shape {}'.format(relu_fc1.get_shape())
    # then a dropout layer
    relu_fc1 = dropout(relu_fc1, keep_prob)

    # then a final fc layer, with output equal to the label count
    fc2_weights = weight_variable([1024, 10])
    fc2_bias = bias_variable([10])
    fc2 = fc(relu_fc1, fc2_weights, fc2_bias)
    print 'fc2 with shape {}'.format(fc2.get_shape())

    return fc2

def next_batch(X_train, y_train, batch_size, total_num, index_in_epoch, epochs_completed,):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > total_num:
        print 'complete {} epochs'.format(epochs_completed)
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(total_num)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
        # start from begining
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= total_num
    end = index_in_epoch
    return X_train[start:end], y_train[start:end], index_in_epoch, epochs_completed




def main():
    # some params
    TEST_SIZE_RATIO = 0.05
    LEARNING_RATE = 1e-4
    TRAINING_ITERATIONS = 20000
    DISPLAY = 10
    DROPOUT = 0.5
    BATCH_SIZE = 50
    SNAPSHOT = 5000
    # get the data
    dh = DataHandler()
    train_X, train_y, test_X = dh.load_data('../dataset/train.csv', '../dataset/test.csv')
    train_X = dh.preprocess(train_X)
    test_X = dh.preprocess(test_X)
    # split the data into train and val

    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, random_state=42, test_size=TEST_SIZE_RATIO)
    # handle the y label to dense to one
    label_count = np.unique(y_train).shape[0]
    y_train = dh.dense_to_one_hot(y_train, label_count)
    y_val = dh.dense_to_one_hot(y_val, label_count)
    # get the width and height of imgs
    img_width = img_height = np.ceil(np.sqrt(X_train.shape[1])).astype(np.uint8)
    img_size = X_train.shape[1]
    num = X_train.shape[0]
    print 'training data size {}'.format(X_train.shape)
    print 'val data size {}'.format(X_val.shape)


    # below is the model
    # the input and output of the NN
    # images
    x = tf.placeholder('float', shape=[None, img_size])
    # labels
    y_ = tf.placeholder('float', shape=[None, label_count])
    # reshape the data
    data = tf.reshape(x, [-1, img_width, img_height, 1])
    keep_prob = tf.placeholder('float')
    fc2 = model(data, keep_prob)
    y = tf.nn.softmax(fc2)
    print 'y with shape {}'.format(y.get_shape())
    # cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # optimization fucntion
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    predict = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    # init
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    # record the accuracy
    train_accuracies = []
    val_accuracies = []
    x_range = []
    saver = tf.train.Saver()
    index_in_epoch = 0
    epochs_completed = 0
    for step in range(TRAINING_ITERATIONS):
        # get the data
        batch_x, batch_y, index_in_epoch, epochs_completed = next_batch(X_train, y_train,
                                                                        BATCH_SIZE, num, index_in_epoch, epochs_completed)
        # check whether to display the accuracy
        if step % DISPLAY == 0 or (step+1) == TRAINING_ITERATIONS:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x,
                                 y_: batch_y,
                                 keep_prob: 1.0})
            val_accuracy = accuracy.eval(feed_dict={x: X_val[0:BATCH_SIZE],
                                y_: y_val[0:BATCH_SIZE],
                                keep_prob: 1.0})
            print 'training accuracy / val accuracy => %.2f/ %.2f for step %d' % (train_accuracy, val_accuracy, step)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            x_range.append(step)
        if step % SNAPSHOT == 0 or (step+1) == TRAINING_ITERATIONS:
            saver.save(sess, 'lenet.ckpt', step)
        # do the training job
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob:DROPOUT})
    # check the hole val-set for accuracy
    val_accuracy = accuracy.eval(feed_dict={x:X_val,y_:y_val,keep_prob:1.0})
    print 'final accuray equals => %.4f' % val_accuracy
    # save those temp files
    with open('train_accuracy.txt', 'wb') as f:
        pickle.dump(train_accuracies, f)
    with open('val_accuracy.txt', 'wb') as f:
        pickle.dump(val_accuracies, f)
    with open('steps.txt', 'wb') as f:
        pickle.dump(x_range, f)
    # do the prediction on test_X
    prediction = np.zeros(test_X.shape[0])
    tnum = test_X.shape[0] / BATCH_SIZE
    for i in range(0, tnum):
        prediction[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = predict.eval(feed_dict={x:test_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], keep_prob: 1.0})
    prediction[tnum*BATCH_SIZE:] = predict.eval(feed_dict={x:test_X[tnum*BATCH_SIZE:], keep_prob:1.0})
    # save to file
    dh.write_data(prediction, '../dataset/cnn_out.csv')


if __name__ == "__main__":
    main()
















