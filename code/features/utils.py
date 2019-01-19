from sklearn.pipeline import FeatureUnion
from features import create_feature
import os
import joblib
from scipy import sparse
import numpy as np


def re_order(order_lst, X, Y):
    X_reordered, Y_reordered = [], []

    def create_dic(X, Y):
        data = {}
        for x, y in zip(X, Y):
            data[x.book_id] = (x, y)
        return data

    mapping = create_dic(X,Y)

    for ele in order_lst:
        X_reordered.append(mapping.get(ele)[0])
        Y_reordered.append(mapping.get(ele)[1])
    return X_reordered, np.array(Y_reordered)


def fetch_features_vectorized(data_dir, features, corpus):
    def extract_feature(feature):

        target_file = os.path.join(data_dir, feature + '.pkl')
        target_index_file = os.path.join(data_dir, feature + '_index.pkl')
        target_labels_file = os.path.join(data_dir, feature + '_labels.pkl')
        target_model_file = os.path.join(data_dir, feature + '_model.pkl')
        if os.path.exists(target_file):
            X_train, X_test = joblib.load(target_file)
            train_books, test_books = joblib.load(target_index_file)

            corpus.X_train,corpus.Y_train=re_order(train_books, corpus.X_train, corpus.Y_train)
            corpus.X_test,corpus.Y_test=re_order(test_books, corpus.X_test, corpus.Y_test)
            for x, x1 in zip(train_books, [book.book_id for book in corpus.X_train]):
                if x != x1:
                    print("Book ids", x, x1)

                    raise AssertionError("Train book order differ")

            for x, x1 in zip(test_books, [book.book_id for book in corpus.X_test]):
                if x != x1:
                    raise AssertionError("Test book order differ")

            #X_train, X_test=corpus.X_train, corpus.X_test
            Y_train, Y_test = corpus.Y_train, corpus.Y_test

        else:
            feature_name, vectorizer = create_feature(feature)

            X_train = vectorizer.fit_transform(corpus.X_train)
            X_test = vectorizer.transform(corpus.X_test)
            joblib.dump((X_train, X_test), target_file)
            joblib.dump((corpus.Y_train, corpus.Y_test), target_labels_file)
            joblib.dump(([book.book_id for book in corpus.X_train], [book.book_id for book in corpus.X_test]),
                        target_index_file)
            joblib.dump(vectorizer, target_model_file)
            Y_train, Y_test = corpus.Y_train, corpus.Y_test
        return X_train, Y_train, X_test, Y_test

    if isinstance(features, list):
        results = [extract_feature(feature) for feature in features]
        # open the results
        X_trains, Y_trains, X_tests, Y_tests = zip(*results)

        if any(sparse.issparse(f) for f in X_trains):
            X_trains = sparse.hstack(X_trains).tocsr()
            X_tests = sparse.hstack(X_tests).tocsr()

        else:
            X_trains = np.hstack(X_trains)
            X_tests = np.hstack(X_tests)

        for i, y in enumerate(Y_trains):
            if i != 0:
                for j, k in zip(y, Y_trains[i - 1]):
                    if j != k:
                        raise AssertionError("Y's differ")
                # assert np.allclose(np.array(y), np.array()), "Y's differ"

                for j, k in zip(Y_tests[i], Y_tests[i - 1]):
                    if j != k:
                        raise AssertionError("Y test's differ")
                        # assert np.allclose(np.array(Y_tests[i]), np.array(Y_tests[i - 1])), "Y test's differ"

        return X_trains, Y_trains[0], X_tests, Y_tests[0]
    else:
        return extract_feature(features)


def test_fetch_features_vectorizer(data_dir, features=['writing_density', 'readability']):
    X_train, Y_train, _, _ = fetch_features_vectorized(data_dir, features, '')
    X_train_w, Y_train_w, _, _ = fetch_features_vectorized(data_dir, 'writing_density', None)
    X_train_r, Y_train_r, _, _ = fetch_features_vectorized(data_dir, 'readability', None)
    assert np.allclose(Y_train, Y_train_r), 'Y does not match'
    assert np.allclose(Y_train, Y_train_w), 'Y does not match'
    assert np.allclose(np.hstack((X_train_w, X_train_r)), X_train), 'Xs does not match'


def johns_features():
    data = np.load('book_datav4_multitask_15_feats.npy')
    data = data.tolist()

    fnames, features, target = data['fnames'], data['features'], data['targets']
    X_train_rnn = []
    X_test_rnn = []
    X_train, X_test = joblib.load('/home/sjmaharjan/Books/booksuccess/vectors/unigram_index.pkl')

    for fname in X_train:
        X_train_rnn.append(features[fnames.index(fname)].tolist())
    for fname in X_test:
        X_test_rnn.append(features[fnames.index(fname)].tolist())

    joblib.dump((np.array([target[fnames.index(fname)] for fname in X_train]),
                 np.array([target[fnames.index(fname)] for fname in X_test])), 'rnn_labels.pk')
