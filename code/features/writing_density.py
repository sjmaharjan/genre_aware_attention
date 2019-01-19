from sklearn.base import BaseEstimator
import re
import numpy as np
import nltk

__all__ = ['paragraphs', 'WritingDensityFeatures']


def paragraphs(document):
    """Helper method to divide document to paragraphs

    Paragraph is defined by punctuation followed by a new line

    Parameters
    ------------
    docuemnt : string


    Retruns
    ----------
    paragraphs : list of paragraphs

    """
    punctuation = '''!"'().?[]`{}'''
    paragraph = re.compile(r'[{}]\n'.format(re.escape(punctuation)))
    return paragraph.split(document)


class WritingDensityFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(
            ['n_words', 'n_chars', 'exclamation', 'question', 'avgwordlenght', 'avesentencelength',
             'avgwordspersentence', 'avgsentperpara','diversity'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        exclamation = [doc.content.count('!') if doc.content else 0 for doc in documents]
        # print exclamation
        question = [doc.content.count('?') if doc.content else 0 for doc in documents]
        # print question

        n_words = [len(nltk.word_tokenize(doc.content)) if doc.content else 0 for doc in documents]
        # print n_words
        n_chars = [len(doc.content) if doc.content else 0 for doc in documents]

        avg_sent_length = [np.mean([len(sent) for sent in nltk.sent_tokenize(doc.content)]) if doc.content else 0 for doc in documents]

        avg_word_lenght = [np.mean([len(word) for word in nltk.word_tokenize(doc.content)]) if doc.content else 0  for doc in documents]

        avg_words_sentence = [np.mean([len(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(doc.content)]) if doc.content else 0 for
                              doc in documents]

        avg_sent_per_paragraph = [np.mean([len(nltk.sent_tokenize(par + ".")) for par in paragraphs(doc.content)]) for
                                  doc in documents]

        diversity = [(len(set(nltk.word_tokenize(doc.content))) * 1.0) / (len(nltk.word_tokenize(doc.content)) * 1.0)
                     if doc.content else 0
                     for doc in
                     documents]

        X = np.array([n_words, n_chars, exclamation, question, avg_word_lenght, avg_sent_length, avg_words_sentence,avg_sent_per_paragraph,
                      diversity]).T
        return X

    def fit_transform(self, documents, y=None):
        return self.transform(documents)

