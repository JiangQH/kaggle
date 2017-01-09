import caffe
import os.path as osp
# current decide to based on the Vgg net
# but change the last fc layer to finetune
model_dir = './model'
net = osp.join(model_dir, 'train.prototxt')
weights = osp.join(model_dir, 'VGG_ILSVRC_16_layers.caffemodel')
