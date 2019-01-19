from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os

__all__ = ['Book2VecFeatures']


class Book2VecFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_dir=None, model_name=None, dtype=np.float32):
        self.model_dir = model_dir
        self.model_name = model_name
        self.dtype = dtype

    def get_feature_names(self):
        return np.array(['book_emb_' + str(i) for i in range(self.num_features_)])

    def make_feature_vec(self, book2vec_id):
        try:
            feature_vec = self.model_.docvecs[book2vec_id]
        except KeyError as e:
            print (e)
            print("Could not fine doc vec for {}".format(book2vec_id))
            feature_vec=np.zeros((self.num_features_,), dtype=self.dtype)

        return feature_vec

    def fit(self, documents, y=None):
        if self.model_name:
            print("Loading Book2Vec")
            model_data = os.path.join(self.model_dir, 'model_%s.doc2vec' % self.model_name)
            if self.model_name == 'dbow_dmm' or self.model_name == 'dbow_dmc':
                m1 = os.path.join(self.model_dir, 'model_%s.doc2vec' % self.model_name.split('_')[0])
                m2 = os.path.join(self.model_dir, 'model_%s.doc2vec' % self.model_name.split('_')[1])
                model1=Doc2Vec.load(m1)
                model2=Doc2Vec.load(m2)
                self.model_ = ConcatenatedDoc2Vec([model1,model2 ])
                self.num_features_ = model1.syn0.shape[1]+model2.syn0.shape[1]
            else:
                self.model_ = Doc2Vec.load(model_data)
                self.num_features_ = self.model_.syn0.shape[1]
            print(self.num_features_)
            print("Done Loading vectors")
        else:
            raise OSError("Model does not exit")

        return self

    def transform(self, documents):
        doc_feature_vecs = np.zeros((len(documents), self.num_features_), dtype=self.dtype)

        for i, doc in enumerate(documents):
            # Print a status message every 1000th review
            if i % 1000. == 0.:
                print("Document %d of %d" % (i, len(documents)))
            #
            #print("Document %s" % doc.book_id)
            doc_feature_vecs[i] = self.make_feature_vec(doc.book2vec_id)

        return doc_feature_vecs
