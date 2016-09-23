# this file use the logistic regression to do the classify job
# there can be preprocessing jobs, like feature-scaling, pca and so on
import pandas as pd
import numpy as np
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import time
import matplotlib.pyplot as plt

# load the data
print('Reading data...')
train = pd.read_csv('../dataSet/train.csv')
test = pd.read_csv('../dataSet/test.csv')
y = train.label.values
X = train.iloc[:, 1:].values
test_x = test.values
print('Load training data ' + str(len(y)) + ', testing data ' + str(len(test_x)))
y = np.array(y)
X = np.array(X)
test_x = np.array(test_x)




# visualize some data
#n = 10
#plt.rcParams["figure.figsize"] = (5, 5)
#f, axes = plt.subplots(n, n)
#plt.tight_layout()
#f.subplots_adjust(hspace=-0.1, wspace=0.1)
#for i in range(n):
#    for j in range(n):
#        axes[i, j].imshow(X[i*n + j].reshape((28,28)), cmap="gray")
#        axes[i, j].axis("off")
#plt.show()

# do pca and logistic regression
# use the sklearn gridsearchcv to set the dimensionality of the pca. and for the regularization for logistic
# sample some data to do the job
print("gridSearchCV for the best regularization term and dimensions of pca...")
X_val = X[:100, :]
y_val = y[:100]
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
# List of (name, transform) tuples (implementing fit/transform) that are chained,
# in the order in which they are chained, with the last object an estimator.
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# plot to visulize
#pca.fit(X_val)
#plt.figure(2, figsize=(4, 3))
#plt.clf()
#plt.axes([.2, .2, .7, .7])
#plt.plot(pca.explained_variance_, linewidth=2)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')

n_components = [20, 40, 60]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator.fit(X_val, y_val)
#plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#            linestyle=':', label='n_components chosen')
#plt.legend(prop=dict(size=12))
#plt.show()
components = estimator.best_estimator_.named_steps['pca'].n_components
cs = estimator.best_estimator_.named_steps['logistic'].C
print ("Estimating with pca components " + str(components) + " and lg cs " + str(cs))
# do the real job
print ("Training ...")
start_time = time.time()
X = estimator.fit(X, y)
print ("Training done with time " + str(time.time()-start_time))
# do the predict job
print ("Predicting ...")
start_time = time.time()
predict = estimator.predict()
print ("Predicting done with time" + str(time.time() - start_time))
# save the prediction
id = range(1, len(predict) + 1)
submissions = pd.DataFrame({'ImageId' : id, 'Label' : predict})
submissions.to_csv('lg_submission.csv', index=False)








