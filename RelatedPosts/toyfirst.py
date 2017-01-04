import os
import os.path as osp
from sklearn.feature_extraction.text import CountVectorizer
import nltk

DIR = './data/toy'
posts = [open(osp.join(DIR, f)).read() for f in
            os.listdir(DIR)]
vectorizer = CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
print 'num_samples {}, num_featrures {}'.format(num_samples, num_features)

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))




