from tools import SimpleTools as ST
from randomFlipImageAndLabelDataLayer import RandomFlipImageAndLabelDataLayerSys
from caffe import layers as L, params as P
import caffe
import os.path as osp
from caffe.proto import caffe_pb2

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
params = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2
model_dir = './model'

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=params, weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=params,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pooling(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX,
                     kernel_size=ks, stride=stride)

# define the net
def vggStyleNet(data_path = None, split='train', batch_size=128, im_shape=(96, 96),
                 selection='all', random_flip=True, from_scratch=False):

    num_output = ST().getOutNum(selection)
    net = caffe.NetSpec()
    param_str = {'batch_size':batch_size, 'path': data_path,
                 'im_shape': im_shape, 'selection': selection, 'random_flip': random_flip}
    python_param = dict(module='randomFlipImageAndLabelDataLayer',
                        layer='RandomFlipImageAndLabelDataLayerSys',
                        param_str=str(param_str))

    if split == 'deploy':
        net.data = L.DummyData(shape=dict(dim=[1, 1, im_shape[0], im_shape[1]]))
    else:
        net.data, net.label = L.Python(python_param=python_param, ntop=2)
    # conv1
    net.conv1_1, net.relu1_1 = conv_relu(net.data, ks=3, nout=64, pad=1)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, ks=3, nout=64, pad=1)
    net.pool1 = max_pooling(net.relu1_2, ks=2, stride=2)

    #conv2
    net.con2_1, net.relu2_1 = conv_relu(net.pool1, ks=3, nout=128, pad=1)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, ks=3, nout=128, pad=1)
    net.pool2 = max_pooling(net.relu2_2, ks=2, stride=2)

    #conv3
    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, ks=3, nout=256, pad=1)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, ks=3, nout=256, pad=1)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, ks=3, nout=256, pad=1)
    net.pool3 = max_pooling(net.relu3_3, ks=2, stride=2)

    #conv4
    #net.conv4_1, net.relu4_1 = conv_relu(net.pool3, ks=3, nout=512, pad=1)
    #net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, ks=3, nout=512, pad=1)
    #net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, ks=3, nout=512, pad=1)
    #net.pool4 = max_pooling(net.relu4_3, ks=2, stride=2)

    #conv5
    #net.conv5_1, net.relu5_1 = conv_relu(net.pool4, ks=3, nout=512, pad=1)
    #net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, ks=3, nout=512, pad=1)
    #net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, ks=3, nout=512, pad=1)
    #net.pool5 = max_pooling(net.relu5_3, ks=2, stride=2)

    # below the self define
    net.fc1, net.relu1 = fc_relu(net.pool3, nout=500)
    if split == 'train':
        net.drop1 = fc2input = L.Dropout(net.relu1, in_place=True)
    else:
        fc2input = net.relu1
    net.fc2, net.relu2 = fc_relu(fc2input, nout=500)
    if split == 'train':
        net.drop2 = fc3input = L.Dropout(net.relu2, in_place=True)
    else:
        fc3input = net.relu2
    net.fc3 = L.InnerProduct(fc3input, num_output=num_output, param=params)
    # test and val have the same loss
    if not split == 'deploy':
        net.loss = L.EuclideanLoss(net.fc3, net.label)

    file_name = osp.join(model_dir, split+'_' + selection+'.prototxt')
    with open(file_name, 'w') as f:
        f.write(str(net.to_proto()))
        f.close()
        return f.name

def solver(prefix, train_net_path, test_net_path=None, base_lr=0.001,
           test_interval=1000, test_iter=100, iter_size=1,
           max_iter=100000, type='SGD', lr_policy='step',
           gamma=0.1, stepsize=20000, momentum=0.9,
           weight_decay=5e-4, display=1, snapshot=10000, mode='gpu'):
    s = caffe_pb2.SolverParameter()
    s.train_net = osp.abspath(train_net_path)
    if test_net_path is not None:
        s.test_net.append(osp.abspath(test_net_path))
        s.test_interval = test_interval
        s.test_iter.append(test_iter)

    s.iter_size = iter_size
    s.max_iter = max_iter
    s.type = type
    s.lr_policy = lr_policy
    s.gamma = gamma
    s.stepsize= stepsize
    s.momentum = momentum
    s.weight_decay = weight_decay
    s.display = display
    s.snapshot = snapshot
    s.snapshot_prefix = prefix + '_facial_keypoint'
    if mode is 'gpu':
        s.solver_mode = caffe_pb2.SolverParameter.GPU
    else:
        s.solver_mode = caffe_pb2.SolverParameter.CPU

    save_name = osp.join(model_dir, prefix+'_solver.prototxt')
    with open(save_name, 'w') as f:
        f.write(str(s))
        f.close()
        return f.name









