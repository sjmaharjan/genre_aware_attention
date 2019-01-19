import os
import nltk
from gensim.models import Word2Vec
import codecs

class GutenbergSentences(object):
    '''
    Tokenizes sentences into words usin nltk
    yields tokens
    '''
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):

        for fid in sorted(os.listdir(self.dirname)):
            fname = os.path.join(self.dirname, fid)
            if fid.startswith('.DS_Store') or not os.path.isfile(fname):
                continue
            with codecs.open(fname, mode='r', encoding='latin1') as infile:
                content = "".join([line.replace('\r\n', '') for line in infile.readlines()])
                if not content:  # don't bother sending out empty sentences
                    continue
                print('%s' % fid)
                for sentence in nltk.sent_tokenize(content):
                    yield nltk.word_tokenize(sentence.lower())


def word2vec_train(corpus_dir, model_dir):
    '''
    Helper function to train word2vec using gensim
    :param corpus_dir: path to list of files
    :param model_dir: output directory where the embdedding are dumped
    :return: None
    '''
    corpus = list(sent for sent in GutenbergSentences(dirname=corpus_dir))
    print("Training")
    model = Word2Vec(corpus, size=300, window=5, min_count=2, workers=40,iter=50, alpha=0.05,negative=25,sg=0)
    model.save_word2vec_format(os.path.join(model_dir, 'gutenbergword2vec.bin'))
    model.save_word2vec_format(os.path.join(model_dir, 'gutenbergword2vec.txt'),binary=False)
    print("Done")
