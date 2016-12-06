#!/usr/bin/env python
# coding=utf-8
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# load the diabetes datasets and use only one feature
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
# split it into training and test
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# create the linear model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# the Coefficients
print('Coefficients: \n', regr.coef_)
# the mean error
print("Residual sum of squares: %.2f"
    % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# explained variance score: 1 is perfect prediction
print ("Variance score: %.2f" % regr.score(diabetes_X_test, diabetes_y_test))

# plot the output
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test),
        color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
