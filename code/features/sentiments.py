from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, OrderedDict
import csv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os

__all__ = ['SentiWordNetFeature', 'SenticConceptsTfidfVectorizer', 'SenticConceptsScores']


# REF http://sentiwordnet.isti.cnr.it/code/SentiWordNetDemoCode.java
# REF Building Machine Learning Systems with Python Section Sentiment analysis
def load_sentiwordnet(path):
    scores = defaultdict(list)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            # skip comments
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            # print POS,PosScore,NegScore,SynsetTerms
            for term in SynsetTerms.split(" "):
                # drop number at the end of every term
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                scores[key].append((float(PosScore), float(NegScore)))
    for key, value in scores.items():
        scores[key] = np.mean(value, axis=0)
    return scores


# REF Building Machine Learning Systems with Python Section Sentiment analysis
class SentiWordNetFeature(BaseEstimator, TransformerMixin):
    __resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self):
        self.sentiwordnet = load_sentiwordnet(os.path.join(self.__resource_dir, 'SentiWordNet_3.0.0_20130122.txt'))

    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def _get_sentiments(self, d):
        tagged_sent = d.word_pos
        pos_vals = []
        neg_vals = []
        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.
        sent_len = 0
        for sentence in tagged_sent.split('\n'):
            for tag in tagged_sent.split():
                try:
                    w, p = tag.rsplit('/', 1)
                except ValueError as e:
                    print(e)
                    print("error for sentence %s" % tag)
                    w, p = '', ''
                p_val, n_val = 0, 0
                sent_pos_type = None
                if p.startswith("NN"):
                    sent_pos_type = "n"
                    nouns += 1
                elif p.startswith("JJ"):
                    sent_pos_type = "a"
                    adjectives += 1
                elif p.startswith("VB"):
                    sent_pos_type = "v"
                    verbs += 1
                elif p.startswith("RB"):
                    sent_pos_type = "r"
                    adverbs += 1
                if sent_pos_type is not None:
                    sent_word = "%s/%s" % (sent_pos_type, w.lower())
                    if sent_word in self.sentiwordnet:
                        p_val, n_val = self.sentiwordnet[sent_word]
                pos_vals.append(p_val)
                neg_vals.append(n_val)
                sent_len += 1

        l = sent_len
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)
        return [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l,
                adverbs / l]

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        X = np.array([self._get_sentiments(d) for d in documents])
        return X


class SentimentsBatch(BaseEstimator, TransformerMixin):
    """Aggregate Sentiments from each document for DictVectorizer"""

    __resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentiwordnet = load_sentiwordnet(os.path.join(self.__resource_dir, 'SentiWordNet_3.0.0_20130122.txt'))

    def _get_sentiments(self, d):
        """
        list of list

        """
        sentiments = []
        for sentence in d.word_pos.split('\n'):
            pos_vals = []
            neg_vals = []
            nouns = 0.
            adjectives = 0.
            verbs = 0.
            adverbs = 0.
            sent_len = 0
            for tag in sentence.split():

                try:
                    w, p = tag.rsplit('/', 1)
                except ValueError as e:
                    print(e)
                    print("error for sentence %s" % tag)
                    w, p = '', ''
                p_val, n_val = 0, 0
                sent_pos_type = None
                if p.startswith("NN"):
                    sent_pos_type = "n"
                    nouns += 1
                elif p.startswith("JJ"):
                    sent_pos_type = "a"
                    adjectives += 1
                elif p.startswith("VB"):
                    sent_pos_type = "v"
                    verbs += 1
                elif p.startswith("RB"):
                    sent_pos_type = "r"
                    adverbs += 1
                if sent_pos_type is not None:
                    sent_word = "%s/%s" % (sent_pos_type, w.lower())
                    if sent_word in self.sentiwordnet:
                        p_val, n_val = self.sentiwordnet[sent_word]
                pos_vals.append(p_val)
                neg_vals.append(n_val)
                sent_len += 1

            l = 1 if sent_len == 0 else sent_len
            avg_pos_val = np.mean(pos_vals) if pos_vals else 0
            avg_neg_val = np.mean(neg_vals) if neg_vals else 0
            sentiments.append(
                [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l,
                 adverbs / l])
        return sentiments

    def fit(self, x, y=None):
        return self

    def transform(self, books):
        features = []
        for book in books:
            X = np.array(self._get_sentiments(book))
            # print (X.shape)
            if X.shape[0] < self.batch_size:

                data_chunks = np.array_split(range(X.shape[0]), X.shape[0])
            else:
                data_chunks = np.array_split(range(X.shape[0]), self.batch_size)
            feature = OrderedDict()
            # print (data_chunks)
            for i, ids in enumerate(data_chunks):
                mean = np.mean(X[ids], axis=0)
                feature['sent_neut' + '_' + str(i)] = round(mean[0], 3)
                feature['sent_pos' + '_' + str(i)] = round(mean[1], 3)
                feature['sent_neg' + '_' + str(i)] = round(mean[2], 3)
                feature['sent_nouns' + '_' + str(i)] = round(mean[3], 3)
                feature['sent_adj' + '_' + str(i)] = round(mean[4], 3)
                feature['sent_verbs' + '_' + str(i)] = round(mean[5], 3)
                feature['sent_adverbs' + '_' + str(i)] = round(mean[6], 3)

            features.append(feature)
        return features


# Sentic Concepts Features




class SenticConceptsTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.concepts))


class SenticConceptsScores(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(
            ['avg_sensitivity', 'avg_attention', 'avg_pleasantness', 'avg_aptitude', 'avg_polarity',
             #        'max_sensitivity',
             # 'max_attention', 'max_pleasantness', 'max_aptitude', 'max_polarity', 'min_sensitivity', 'min_attention',
             # 'min_pleasantness', 'min_aptitude', 'min_polarity'
             ])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        feature_vector = []
        for doc in documents:
            avg_sensitivity = np.mean(doc.sensitivity) if doc.sensitivity else 0.
            # min_sensitivity = np.min(doc.sensitivity)
            # max_sensitivity = np.max(doc.sensitivity)

            avg_attention = np.mean(doc.attention) if doc.attention  else 0.
            # min_attention = np.min(doc.attention)
            # max_attention = np.max(doc.attention)

            avg_pleasantness = np.mean(doc.pleasantness) if doc.pleasantness else 0.
            # min_pleasantness = np.min(doc.pleasantness)
            # max_pleasantness = np.max(doc.pleasantness)

            avg_aptitude = np.mean(doc.aptitude) if doc.aptitude else 0.
            # min_aptitude = np.min(doc.aptitude)
            # max_aptitude = np.max(doc.aptitude)

            avg_polarity = np.mean(doc.polarity) if doc.polarity  else 0.
            # min_polarity = np.min(doc.polarity)
            # max_polarity = np.max(doc.polarity)

            feature_vector.append(
                [avg_sensitivity, avg_attention, avg_pleasantness, avg_aptitude, avg_polarity,
                 # max_sensitivity,
                 # max_attention, max_pleasantness, max_aptitude, max_aptitude, max_polarity, min_attention,
                 # min_pleasantness, min_aptitude, min_aptitude, min_polarity
                 ])

        return np.array(feature_vector)
