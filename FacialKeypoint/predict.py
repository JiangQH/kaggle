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


def prediction(model, img, save_name=None):
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
    for model in models:
        model_dir = osp.join('./model/or', model)
        net_file = osp.join(model_dir, 'deploy.prototxt')
        weights = osp.join(model_dir, 'weights.caffemodel')
        net = caffe.Net(net_file, weights, caffe.TEST)
        nets[model] = net

    lookup = pandas.read_csv('./data/IdLookupTable.csv')
    rowid = lookup['RowId'].values
    imageid = lookup['ImageId'].values
    feature_name = lookup['FeatureName'].values

    # the prediction
    predictions = {model_name: [] for model_name in models}
    for it in range(len(data)):
        print it
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

    locations = []
    for it in range(len(imageid)):
        im_id = imageid[it] - 1
        fea_name = feature_name[it]
        # get the model and corresponding index
        model_name, index = ST().getModelAndIndex(fea_name)
        locations.append(predictions[model_name][im_id][index])

    # save
    ST().write_prediction(locations, rowid, './data/predict_combine_or.csv')









models = ['all', 'eye_center',  'eye_outer', 'mouth']
test_data = ST().load_data(path='./data/test.csv', phase='Test')
cross_prediction(models, test_data)








