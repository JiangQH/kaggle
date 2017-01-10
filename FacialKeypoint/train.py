import caffe
import os.path as osp
from net import vggStyleNet, solver, normalStyleNet
from tools import SimpleTools as ST
import numpy as np
# current decide to based on the Vgg net
# but change the last fc layer to finetune
model_dir = './model'

def train(selection='all', style='Vgg'):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    data_dir = './data'
    # the whole net work
    select = selection
    print '----split the data set, train the whole net----'
    #ST().splitTrainingData(osp.join(data_dir, 'training.csv'), select=select)

    if style == 'vgg':
        train = vggStyleNet(osp.join(data_dir, select+'_train.hdf5'), split='train',
                            batch_size=128, im_shape=(96, 96), selection=select,
                            random_flip=True)
        val = vggStyleNet(osp.join(data_dir, select+'_val.hdf5'), split='val',
                        batch_size=128, im_shape=(96, 96), selection=select,
                        random_flip=False)
        solver_file = caffe.get_solver(solver(prefix=select, train_net_path=train, test_net_path=val,
                        mode='gpu', base_lr=0))

    else:
        train = normalStyleNet(osp.join(data_dir, select + '_train.hdf5'), split='train',
                            batch_size=1, im_shape=(96, 96), selection=select,
                            random_flip=True)
        val = normalStyleNet(osp.join(data_dir, select + '_val.hdf5'), split='val',
                          batch_size=1, im_shape=(96, 96), selection=select,
                          random_flip=False)
        solver_file = caffe.get_solver(solver(prefix=select, train_net_path=train, test_net_path=val,
                                              mode='gpu', base_lr=0))



    niter = 30000
    train_loss = np.zeros(niter)
    test_interval = 100
    test_iter = 2
    val_loss = np.zeros(int(np.ceil(niter / test_interval)))
    display = 10
    snapshot = 10000
    for it in range(niter):
        solver_file.step(1)
        train_loss[it] = solver_file.net.blobs['loss'].data
        if it % display == 0 or it + 1 == niter:
            print 'Iteration {} with loss {}'.format(it, train_loss[it])
        if it % test_interval == 0:
            test_loss = 0
            for test_it in range(test_iter):
                solver_file.test_nets[0].forward()
                test_loss += solver_file.test_nets[0].blobs['loss'].data
            val_loss[it // test_interval] = test_loss/test_iter
            print 'Iteration {} the val score ... {}'.format(it, test_loss/test_iter)

        if (it+1) % snapshot == 0:
            filename = osp.join(model_dir, 'weights_{}.{}.caffemodel'.format(style, it))
            solver_file.net.save(filename)

    np.savetxt('{}_trainlog.txt'.format(style), train_loss)
    np.savetxt('{}_vallog.txt'.format(style), val_loss)


train(style='vgg')


















