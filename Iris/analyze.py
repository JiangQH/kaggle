import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_name = data['target_names']
#plt.figure(figsize=(10, 15))
fig, axes = plt.subplots(2, 3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

for m, (i, j) in enumerate(pairs):
    ax = axes.flat[m]
    for t, marker, c in zip(xrange(3), ('>', 'o', 'x'), ('r', 'g', 'b')):
        ax.scatter(features[target == t, i],
                   features[target == t, j],
                   marker=marker,
                   c=c)
    ax.set_xlabel("{}".format(feature_names[i]))
    ax.set_ylabel("{}".format(feature_names[j]))
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.show()
#fig.savefig('data_feature.png')


# maybe we can seperate the setosa from others by the petal length
plength = features[:, 2]
is_setosa = target_name[target] == 'setosa'
max_plength_setosa = plength[is_setosa].max()
min_plength_others = plength[~is_setosa].min()
print 'max of setosa petal length {}'.format(max_plength_setosa)
print 'min of setosa petal length {}'.format(min_plength_others)
# we can achieve perfect to seperate the setosa from others using only the petal length.
# now go further

features = features[~is_setosa]
labels = target_name[target]
labels = labels[~is_setosa]

virginica = (labels == 'virginica')







