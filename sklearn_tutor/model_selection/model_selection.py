#!/usr/bin/env python
# coding=utf-8
from sklearn import svm, datasets
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
# kfold cross validation
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print scores

# cross-validation generators, using from sklearn
from sklearn import cross_validation
kfold = cross_validation.KFold(len(X_digits), n_folds=3)
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test],
                                                y_digits[test])
        for train, test in kfold]

# use the helper to do score
cross_validation.cross_val_score(svc, X_digits, y_digits, 
                                cv=kfold, n_jobs=-1)


