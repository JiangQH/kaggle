from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
                'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
        categories=categories, shuffle=True, random_state=42)

# extracting features from the txt(bag of words)
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(twenty_train.data)
print(X_train_counts.shape)
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# training a classifier
# first using the naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is love', 'OPenGL on the GPU is fast']
X_new_counts = count_vec.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print(' %r => %s' % (doc, twenty_train.target_names[category]))

# building a pipeline
# in order to make the vectorizer-->transformer-->classifier easier to use
from sklearn.pipeline import Pipeline
txt_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                    ])
txt_clf.fit(twenty_train.data, twenty_train.target)

#evaluation of the performance on the test set
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
        categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = txt_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#using svm instead
from sklearn.linear_model import SGDClassifier
txt_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='12',
                                            alpha=1e-3, n_iter=5, random_state=42)),
                    ])
txt_clf.fit(twenty_train.data, twenty_train.target)
predicted = txt_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))

## parameter tuning using grid search
from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range', [(1, 1), (1, 2)],
                'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(txt_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
















