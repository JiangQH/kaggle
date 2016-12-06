from __future__ import print_function
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores = list()

for alpha in alphas:
	lasso.alpha = alpha
	this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=-1)
	scores.append(np.mean(this_scores))

plt.figure(figsize=(4, 3))
plt.semilogx(alphas, scores)
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.show()
