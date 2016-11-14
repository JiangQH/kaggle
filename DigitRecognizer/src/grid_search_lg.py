from tools import DataHandler
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
from time import time

# load the data
dh = DataHandler()
train_X, train_y, test_X = dh.load_data('../dataset/train.csv', '../dataset/test.csv')
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, random_state=42, test_size=0.1)

# to visualize?
# dh.vis_data(train_X, train_y)

# a pca and the classifier
pca = PCA()
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=150, multi_class='multinomial', n_jobs=-1)


# the pipeline
pipe = Pipeline(steps=[('pca', pca), ('lg', logistic)])


# grid search for the best params for pca(ncomponents, whiten), logistic(C,)
ncomponents = [30, 60, 90, 120]
Cs = np.logspace(-2, 2, 5)
whiten = [True, False]
params = {
    'pca__whiten': whiten,
    'pca__n_components': ncomponents,
    'lg__C': Cs,
}

# do the search work
grid_search = GridSearchCV(pipe, params, n_jobs=-1, verbose=1)
print ("Performing grid searching...")
t0 = time()
grid_search.fit(X_train, y_train)
print ("done gridsearch with score %0.3f" % grid_search.best_score_)
best_params = grid_search.best_params_
for param_name in sorted(params):
    print ("\t%s: %r" %(param_name, best_params[param_name]))

# print the actuall score
score = grid_search.score(X_val, y_val)
print ("Real prediction score is: {}".format(score))

# do the prediction
prediction = grid_search.predict(test_X)
dh.write_data(prediction, '../dataset/out.csv')