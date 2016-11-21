from tools import DataHandler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


dh = DataHandler()
train_X, train_y, test_X = dh.load_data('../dataset/train.csv', '../dataset/test.csv')
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, random_state=42, test_size=0.1)
#dh.vis_data(train_X, train_y)

# add preprocessing , pca?
ncomponets = 100
pca = PCA(n_components=ncomponets, whiten=True)
print "begin fit pca"
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# do the logistic regression?
logistic = linear_model.LogisticRegression(solver='sag', max_iter=100, multi_class='multinomial', n_jobs=-1)
print "begin fitting"
logistic.fit(X_train, y_train)
score = logistic.score(X_test, y_test)
print "with score {}".format(score)

# prediction the test data
test_X = pca.transform(test_X)
prediction = logistic.predict(test_X)
# write to the file
dh.write_data(prediction, '../dataset/out.csv')


