import caffe
import os.path as osp
from net import vggStyleNet, solver
from tools import SimpleTools as ST
# current decide to based on the Vgg net
# but change the last fc layer to finetune


model_dir = './model'
data_dir = './data'
# the whole net work
select = 'all'
if not osp.exists(osp.join(data_dir, select+'_train.hdf5')) or not osp.exists(osp.join(data_dir, select+'_val.hdf5')):
    print '----split the data set----'
    ST().splitTrainingData(osp.join(data_dir, 'training.csv'), select=select)


whole_train = vggStyleNet(osp.join(data_dir, select+'_train.hdf5'), split='train',
                    batch_size=1, im_shape=(96, 96), selection=select,
                    random_flip=True)

whole_val = vggStyleNet(osp.join(data_dir, select+'_val.hdf5'), split='val',
                  batch_size=1, im_shape=(96, 96), selection=select,
                  random_flip=False)
whole_solver = caffe.get_solver(solver(prefix=select, train_net_path=whole_train, test_net_path=whole_val,
                      mode='gpu', test_iter=5, base_lr=0))
#weights = osp.join(model_dir, 'VGG_ILSVRC_16_layers.caffemodel')
#whole_solver.net.copy_from(weights)
niter = 100000
for it in range(niter):
    whole_solver.step(1)
    im = whole_solver.net.blobs['data'].data[0, 0, ...]
    label = whole_solver.net.blobs['label'].data[0, :]
    im, label = ST().umcompress(im, label)
    ST().plot(im, label)







