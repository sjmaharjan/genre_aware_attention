import string
import re
import itertools
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np




__all__ = ['NGramTfidfVectorizer','CategoricalCharNgramsVectorizer','KSkipNgramsVectorizer']


class NGramTfidfVectorizer(TfidfVectorizer):
    """Convert a collection of  documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
    """

    def build_analyzer(self):
        """Overrides the super class method

        Parameter
        ----------
        self

        Returns
        ----------
        analyzer : function
            extract content from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))


class CategoricalCharNgramsVectorizer(TfidfVectorizer):
    """ Typed character ngrams
    Refer to http://www.aclweb.org/anthology/N15-1010

    """

    _slash_W = string.punctuation + " "

    _punctuation = r'''['\"“”‘’.?!…,:;#\<\=\>@\(\)\*-]'''
    _beg_punct = lambda self, x: re.match('^' + self._punctuation + '\w+', x)
    _mid_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '(?:\w+|\s+)', x)
    _end_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '$', x)

    # re.match is anchored at the beginning
    _whole_word = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        i == 0 or y[i - 1] in self._slash_W) and (i + n == len(y) or y[i + n] in self._slash_W)
    _mid_word = lambda self, x, y, i, n: not (
        re.findall(r'(?:\W|\s)', x) or i == 0 or y[i - 1] in self._slash_W or i + n == len(y) or y[
            i + n] in self._slash_W)
    _multi_word = lambda self, x: re.match('\w+\s\w+', x)

    _prefix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (i == 0 or y[i - 1] in self._slash_W) and (
        not (i + n == len(y) or y[i + n] in self._slash_W))
    _suffix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        not (i == 0 or y[i - 1] in self._slash_W)) and (i + n == len(y) or y[i + n] in self._slash_W)
    _space_prefix = lambda self, x: re.match(r'''^\s\w+''', x)
    _space_suffix = lambda self, x: re.match(r'''\w+\s$''', x)

    def __init__(self, beg_punct=None, mid_punct=None, end_punct=None, whole_word=None, mid_word=None, multi_word=None,
                 prefix=None, suffix=None, space_prefix=None, space_suffix=None, all=None, input='content',
                 encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        # super(CategoricalCharNgramsVectorizer, self).__init__(**kwargs)
        self.beg_punct = beg_punct
        self.mid_punct = mid_punct
        self.end_punct = end_punct
        self.whole_word = whole_word
        self.mid_word = mid_word
        self.multi_word = multi_word
        self.prefix = prefix
        self.suffix = suffix
        self.space_prefix = space_prefix
        self.space_suffix = space_suffix
        self.all = all
        if self.all:
            print('here categorical ....2')
            self.beg_punct = True
            self.mid_punct = True
            self.end_punct = True
            self.whole_word = True
            self.mid_word = True
            self.multi_word = True
            self.prefix = True
            self.suffix = True
            self.space_prefix = True
            self.space_suffix = True
        super(CategoricalCharNgramsVectorizer, self).__init__(input=input, encoding=encoding, decode_error=decode_error,
                                                              strip_accents=strip_accents, lowercase=lowercase,
                                                              preprocessor=preprocessor, tokenizer=tokenizer,
                                                              analyzer=analyzer,
                                                              stop_words=stop_words, token_pattern=token_pattern,
                                                              ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                                              max_features=max_features, vocabulary=vocabulary,
                                                              binary=binary,
                                                              dtype=dtype)

    def _categorical_char_ngrams(self, text_document):
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        # print min_n,max_n
        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                # check categories
                gram = text_document[i: i + n]
                added = False

                # punctuations
                if self.beg_punct and not added:
                    if self._beg_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.mid_punct and not added:
                    if self._mid_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.end_punct and not added:
                    if self._end_punct(gram):
                        ngrams.append(gram)
                        added = True

                # words


                if self.multi_word and not added:
                    if self._multi_word(gram):
                        ngrams.append(gram)
                        added = True

                if self.whole_word and not added:
                    if self._whole_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.mid_word and not added:
                    if self._mid_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                # affixes
                if self.space_prefix and not added:
                    if self._space_prefix(gram):
                        ngrams.append(gram)
                        added = True

                if self.space_suffix and not added:
                    if self._space_suffix(gram):
                        ngrams.append(gram)
                        added = True

                if self.prefix and not added:
                    if self._prefix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.suffix and not added:
                    if self._suffix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

        return ngrams

    def build_analyzer(self):
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._categorical_char_ngrams(preprocess(doc.content))






class KSkipNgramsVectorizer(TfidfVectorizer):
    """ k skip n gram Feature estimator
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf

    """

    def __init__(self, k=1, ngram=2, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.ngram = ngram
        super(KSkipNgramsVectorizer, self).__init__(input=input, encoding=encoding, decode_error=decode_error,
                                                    strip_accents=strip_accents, lowercase=lowercase,
                                                    preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                                                    stop_words=stop_words, token_pattern=token_pattern,
                                                    ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                                    max_features=max_features, vocabulary=vocabulary, binary=binary,
                                                    dtype=dtype)

    def _skip_grams_sentence(self, sentence, stop_words=None):
        tokens = sentence
        if stop_words is not None:
            tokens = [w for w in sentence if w not in stop_words]

        k, ngram = self.k, self.ngram
        original_tokens = tokens
        n_original_tokens = len(original_tokens)

        skip_grams = []
        for i in range(n_original_tokens - ngram + 1):
            for x in itertools.combinations(original_tokens[i:i + k + ngram], ngram):
                skip_grams.append(x)
        return [" ".join(skip_gram) for skip_gram in sorted(set(skip_grams))]

    def _k_skip_ngrams(self, text_document, stop_words):
        tokenize = super(TfidfVectorizer, self).build_tokenizer()
        tokens = []
        for sentence in nltk.sent_tokenize(text_document):
            tokens += self._skip_grams_sentence(tokenize(sentence), stop_words)
        return tokens

    def build_analyzer(self):
        stop_words = super(TfidfVectorizer, self).get_stop_words()
        tokenize = super(TfidfVectorizer, self).build_tokenizer()
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._k_skip_ngrams(preprocess(self.decode(doc.content)), stop_words)