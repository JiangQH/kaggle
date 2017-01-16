import caffe
import os.path as osp
from tools import SimpleTools as ST
import pandas
"""
lookup = pandas.read_csv('./data/IdLookupTable.csv')
rowid = lookup['RowId'].values
imageid = lookup['ImageId'].values
featurename = lookup[lookup.columns[2:-1]].values[:, 0]
featureIndex = ST().getIndex(featurename)
model_dir = './model/or/eye_center'
net_file = osp.join(model_dir, 'deploy.prototxt')
weights = osp.join(model_dir, 'or__iter_30000.caffemodel')
caffe.set_mode_cpu()
net = caffe.Net(net_file, weights, caffe.TEST)

# load the test data

test_data = ST().load_data(path='./data/test.csv', phase='Test')
predictions = []
for it in range(len(test_data)):
    print it
    X = test_data[10, :, :]
    X = ST().preprocess(X)
    net.blobs['data'].data[...] = X
    net.forward()
    prediction = net.blobs['fc6_fine'].data[0, ...]
    prediction = ST().getTruey(prediction)
    ST().plot(X, prediction)
    predictions.append(prediction)
# get the value for prediction
locations = [predictions[i-1][j] for i, j in zip(imageid, featureIndex)]
ST().write_prediction(locations, rowid, './data/predict_or.csv')
print 'done'
"""


def prediction(model, save_name=None):
    model_dir = osp.join('./model/or', model)
    net_file = osp.join(model_dir, 'deploy.prototxt')
    weights = osp.join(model_dir, 'weights.caffemodel')
    caffe.set_mode_cpu()
    net = caffe.Net(net_file, weights, caffe.TEST)
    X = ST().preprocess(img)
    net.blobs['data'].data[...] = X
    net.forward()
    if model != 'all':
        prediction = net.blobs['fc6_fine'].data[0, ...]
    else:
        prediction = net.blobs['fc6'].data[0, ...]
    prediction = ST().getTruey(prediction)
    ST().plot(X, prediction, save_name)


def cross_prediction(models, data):
    caffe.set_mode_cpu()
    nets = {}
    featureindex = {}
    for model in models:
        model_dir = osp.join('./model/or', model)
        net_file = osp.join(model_dir, 'deploy.prototxt')
        weights = osp.join(model_dir, 'weights.caffemodel')
        net = caffe.Net(net_file, weights, caffe.TEST)
        nets[model] = net
        featurename = ST().getFeatureName(model)
        featureindex[model] = ST().getIndex(featurename)

    lookup = pandas.read_csv('./data/IdLookupTable.csv')
    rowid = lookup['RowId'].valuesrowid
    imageid = lookup['ImageId'].valuesimageid

    # the prediction
    predictions = {model_name: [] for model_name in models}
    for it in range(len(data)):
        X = data[it, :, :]
        X = ST().preprocess(X)
        for key in nets:
            nets[key].blobs['data'].data[...] = X
            nets[key].forward()
            try:
                pre = nets[key].blobs['fc6_fine'].data[0, ...]
            except:
                pre = nets[key].blobs['fc6'].data[0, ...]
            pre = ST().getTruey(pre)
            predictions[key].append(pre)

    # having traveled, the prediction for all
    all_locations = [predictions['all'][i-1][j] for i, j in zip(imageid, featureindex['all'])]
    # then fill others
    for model in models:
        if model == 'all':
            continue
        model_pre = [predictions[model][i-1][j] for i, j in zip(imageid, featureindex[model])]
        # replace the index
        all_locations[:, featureindex[model]] = model_pre

    # save
    ST().write_prediction(all_locations, rowid, './data/predict_combine_or.csv')









models = ['eye_center',  'eye_outer', 'mouth']
test_data = ST().load_data(path='./data/test.csv', phase='Test')
for model in models:
    img = test_data[0, :, :]
    save_name = osp.join('./result/', model+'2.png')
    prediction(model, img, save_name)








