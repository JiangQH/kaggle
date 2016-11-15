from tools import DataHandler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from time import time

dh = DataHandler()
train_X, train_y, test_X = dh.load_data('../dataset/train.csv', '../dataset/test.csv')
train_data, val_data, train_label, val_label = train_test_split(train_X, train_y, random_state=42, test_size=0.1)

# construct a pipeline to to select the best params
pca = PCA()
svc = SVC()
ncomponents = [30, 60, 90, 120]
Cs = np.logspace(-4,4,5)
whiten = [True, False]
pipe = Pipeline(steps=[('pca', pca), ('svm', svc)])
params = {
    'pca__n_components': ncomponents,
    'pca__whiten': whiten,
    'svm__C': Cs,
}

# train the grid search
estimator = GridSearchCV(pipe, params, n_jobs=-1, verbose=0)
print "begin training the grid search "
t0 = time()
estimator.fit(train_data, train_label)
print "fitting done with {} seconds".format(time()-t0)
print "grid search with best score {}".format(estimator.best_score_)
best_params = estimator.best_params_
for param_name in sorted(params):
    print "{} : {}".format(param_name, best_params[param_name])

# give the actually test score
score = estimator.score(val_data, val_label)
print "With score {}".format(score)

# do the prediction and save to file
prediction = estimator.predict(test_X)
dh.write_data(prediction, '../dataset/svm_out.csv')