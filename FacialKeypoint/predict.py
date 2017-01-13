import caffe
import os.path as osp
from tools import SimpleTools as ST
import pandas

lookup = pandas.read_csv('./data/IdLookupTable.csv')
rowid = lookup['RowId'].values
imageid = lookup['ImageId'].values
featurename = lookup[lookup.columns[2:-1]].values[:, 0]
featureIndex = ST().getIndex(featurename)


model_dir = './model/or'
net_file = osp.join(model_dir, 'or_deploy_all.prototxt')
weights = osp.join(model_dir, 'or__iter_30000.caffemodel')
caffe.set_mode_cpu()
net = caffe.Net(net_file, weights, caffe.TEST)

# load the test data

test_data = ST().load_data(path='./data/test.csv', phase='Test')
predictions = []
for it in range(len(test_data)):
    print it
    X = test_data[it, :, :]
    X = ST().preprocess(X)
    net.blobs['data'].data[...] = X
    net.forward()
    prediction = net.blobs['fc6'].data[0, ...]
    prediction = ST().getTruey(prediction)
    #ST().plot(X, prediction)
    predictions.append(prediction)
# get the value for prediction
locations = [predictions[i-1][j] for i, j in zip(imageid, featureIndex)]
ST().write_prediction(locations, rowid, './data/predict_or.csv')
print 'done'





