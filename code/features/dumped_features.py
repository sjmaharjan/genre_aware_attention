from sklearn.base import BaseEstimator, TransformerMixin
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from manage import app

__all__ = ['DumpedFeaturesTransformers', 'BookCoverFeature']


class DumpedFeaturesTransformers(BaseEstimator, TransformerMixin):
    """
    Loads the dumped features

    """
    # __dumped_dir = current_app.config['VECTORS']
    __dumped_dir = app.VECTORS

    def __init__(self, feature):
        self.feature = feature

        if os.path.exists(os.path.join(self.__dumped_dir, self.feature + '.vector')):
            # self._X_ids = fvs.feature_vectors[self.feature].ids()
            # self._vectors = fvs.feature_vectors[self.feature].vectors()
            # self._model = fvs.feature_vectors[self.feature].model()
            pass

        else:
            raise ValueError("Feature dump for  %s does not exist in %s" % (
                feature, os.path.join(self.__dumped_dir, feature + '.vector')))

    def get_feature_names(self):
        return self._model.get_feature_names()

    def fit(self, X, y=None):
        return self

    def transform(self, books):
        X = []
        sparse = sp.issparse(self._vectors)
        for book in books:

            if book.book_id in self._X_ids:
                book_index = self._X_ids.index(book.book_id)
                if sparse:
                    X.append(self._vectors[book_index].toarray()[0])
                else:
                    X.append(self._vectors[book_index])
            else:
                # this should not happen
                print("Herer inside danger zone")
                X.append(self._model.transform(book)[0])

        if sparse:
            # print X[0]

            X = sp.csr_matrix(X)
        else:
            X = np.array(X)
        # print X[0]
        return X


class BookCoverFeature(BaseEstimator, TransformerMixin):
    '''
    Alex net from Pastor as feature extractor
    Loads the book cover features dumped in file

    '''

    def __init__(self, feature_dump, norm=None):
        self.feature_loc = feature_dump
        self.vectors = self._load()
        self.norm = norm
        self.num_features_ = len(self.vectors.columns)

    def _load(self):
        df = pd.read_csv(self.feature_loc, encoding='utf-8', header=None)
        for img_type in ['jpg','jpeg','png']:
            df[0] = df[0].apply(
                lambda x: os.path.basename(x).replace('.'+img_type, '.txt').replace('-', '+'))  # get the file name
        df.set_index(0, inplace=True)

        return df

    def fit(self, X, y=None):
        return self

    def transform(self, books):
        X = []

        for book in books:
            try:
                feature_vector = self.vectors.loc[book.book_id, :].values
                if feature_vector.shape[0] != self.num_features_:
                    print("ERROR!!!")
                    print (book.book_id)
                    print (feature_vector.shape)
                    raise(ValueError('Vector shape different'))
            except KeyError as e:
                print("Could not fine book cover image vec for {}".format(book.book_id))
                feature_vector = np.zeros((self.num_features_,), dtype=np.float32)
            X.append(feature_vector)

        X = np.array(X)


        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X
