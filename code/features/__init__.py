import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
import nltk
from sklearn.preprocessing import StandardScaler

from . import lexical
from . import syntactic
from . import word_embeddings
from . import phonetic
from . import readability
from . import sentiments
from . import writing_density
from . import dumped_features
from . import book2vec
from manage import app





# __all__ = ['lexical', 'word_embeddings', 'phonetic', 'readability', 'writing_density', 'sentiments', 'get_feature',
#            'create_feature', 'dumped_features', 'book2vec']

__all__ = ['lexical', 'word_embeddings', 'phonetic', 'writing_density', 'sentiments', 'get_feature',
           'create_feature', 'dumped_features', 'new_features','book2vec']

def preprocess(x):
    return x.lower()


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(
        unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                             lowercase=True, min_df=2),
        bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2),  preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                            lowercase=True, min_df=2),
        trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3),  preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                             lowercase=True, min_df=2),

        #
        # #char ngram
        char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), preprocessor=preprocess,analyzer="char",
                                              lowercase=True, min_df=2),
        char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4),preprocessor=preprocess, analyzer="char", lowercase=True, min_df=2),

        char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5),preprocessor=preprocess, analyzer="char", lowercase=True, min_df=2),

        # categorical character ngrams
        categorical_char_ngram_beg_punct=lexical.CategoricalCharNgramsVectorizer(beg_punct=True,preprocessor=preprocess, ngram_range=(3, 3)),
        categorical_char_ngram_mid_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, mid_punct=True),
        categorical_char_ngram_end_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, end_punct=True),

        categorical_char_ngram_multi_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, multi_word=True),
        categorical_char_ngram_whole_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, whole_word=True),
        categorical_char_ngram_mid_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, mid_word=True),

        categorical_char_ngram_space_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess,
                                                                                    space_prefix=True),
        categorical_char_ngram_space_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),
                                                                                    space_suffix=True),

        categorical_char_ngram_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), preprocessor=preprocess,prefix=True),
        categorical_char_ngram_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, suffix=True),

        # #skip gram
        two_skip_3_grams=lexical.KSkipNgramsVectorizer(k=2, ngram=3,preprocessor=preprocess, lowercase=True),
        two_skip_2_grams=lexical.KSkipNgramsVectorizer(k=2, ngram=2,preprocessor=preprocess, lowercase=True),

        # pos
        pos=syntactic.POSTags(ngram_range=(1, 1), tokenizer=str.split, analyzer="word", use_idf=False, norm='l1'),

        # #phrasal and clausal
        phrasal=syntactic.Constituents(PHR=True),
        clausal=syntactic.Constituents(CLS=True),
        phr_cls=syntactic.Constituents(PHR=True, CLS=True),

        # #lexicalized and unlexicalized production rules
        lexicalized=syntactic.LexicalizedProduction(use_idf=False),
        unlexicalized=syntactic.UnLexicalizedProduction(use_idf=False),
        gp_lexicalized=syntactic.GrandParentLexicalizedProduction(use_idf=False),
        gp_unlexicalized=syntactic.GrandParentUnLexicalizedProduction(use_idf=False),

        # writing density
        writing_density=writing_density.WritingDensityFeatures(),
        writing_density_scaled=Pipeline([('wr',writing_density.WritingDensityFeatures()),('scaler',StandardScaler(with_mean=False))]),


        # readability
        #readability=readability.ReadabilityIndicesFeatures(),
        # readability_scaled=Pipeline([('rd',readability.ReadabilityIndicesFeatures()),('scaler',StandardScaler(with_mean=False))]),

        concepts=sentiments.SenticConceptsTfidfVectorizer(ngram_range=(1, 1), tokenizer=str.split, analyzer="word",
                                                          lowercase=True, binary=True, use_idf=False),

        concepts_score=sentiments.SenticConceptsScores(),

        google_word_emb=word_embeddings.Word2VecFeatures(tokenizer=nltk.word_tokenize, analyzer="word",
                                                    lowercase=True,
                                                    model_name=app.GOOGLE_EMB),

        gutenberg_word_emb=word_embeddings.Word2VecFeatures(tokenizer=nltk.word_tokenize, analyzer="word",
                                                         lowercase=True,
                                                         model_name=app.GUTENBERG_EMB),

        gutenberg_cbow_word_emb=word_embeddings.Word2VecFeatures(tokenizer=nltk.word_tokenize, analyzer="word",
                                                            lowercase=True,
                                                            model_name=app.GUTENBERG_EMB),


        #book2Vec dmc', 'dbow', 'dmm', 'dbow_dmm', 'dbow_dmc'

        book2vec_dmc=book2vec.Book2VecFeatures(model_dir=app.BOOK2VEC,model_name='dmc'),
        book2vec_dbow=book2vec.Book2VecFeatures(model_dir=app.BOOK2VEC,model_name='dbow'),
        book2vec_dmm=book2vec.Book2VecFeatures(model_dir=app.BOOK2VEC,model_name='dmm'),
        book2vec_dbow_dmm=book2vec.Book2VecFeatures(model_dir=app.BOOK2VEC,model_name='dbow_dmm'),
        book2vec_dbow_dmc=book2vec.Book2VecFeatures(model_dir=app.BOOK2VEC,model_name='dbow_dmc'),

        #book1002vec
        book10002vec_dmc=book2vec.Book2VecFeatures(model_dir=app.BOOK10002VEC,model_name='dmc'),
        book10002vec_dbow=book2vec.Book2VecFeatures(model_dir=app.BOOK10002VEC,model_name='dbow'),
        book10002vec_dmm=book2vec.Book2VecFeatures(model_dir=app.BOOK10002VEC,model_name='dmm'),
        book10002vec_dbow_dmm=book2vec.Book2VecFeatures(model_dir=app.BOOK10002VEC,model_name='dbow_dmm'),
        book10002vec_dbow_dmc=book2vec.Book2VecFeatures(model_dir=app.BOOK10002VEC,model_name='dbow_dmc'),


        # phonetics
        phonetic=phonetic.PhoneticCharNgramsVectorizer(ngram_range=(3, 3), analyzer='char', min_df=2, lowercase=False),

        phonetic_scores=phonetic.PhonemeGroupBasedFeatures(),

        # sentiWordNet
        swn = sentiments.SentiWordNetFeature(),
        swn_batch = Pipeline([('sent_avb',sentiments.SentimentsBatch(batch_size=50)),('vec', DictVectorizer(sparse=False))]),


        #stress patterns
        stress_ngrams=phonetic.StressNgramsVectorizer(ngram_range=(3, 3), min_df=2),
        stress_scores=phonetic.StressFeatures(),

        ##visual features

        alex_net=dumped_features.BookCoverFeature(feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/log.txt'),
        alex_net_norm=dumped_features.BookCoverFeature(feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/log.txt', norm='l2'),
        vgg16=dumped_features.BookCoverFeature(
            feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/vgg_book_cover_features_2.csv'),
        vgg16_norm=dumped_features.BookCoverFeature(
            feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/vgg_book_cover_features_2.csv', norm='l2'),

        resnet50=dumped_features.BookCoverFeature(
            feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/resnet_book_cover_features_2.csv'),
        resnet50_norm=dumped_features.BookCoverFeature(
            feature_dump='/uhpc/solorio/suraj/Books/book_cover_features/resnet_book_cover_features_2.csv',
            norm='l2'),





    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """
    try:
        #print (feature_names)
        if isinstance(feature_names, list):
            return ("-".join(feature_names), FeatureUnion([(f, get_feature(f)) for f in feature_names]))
        else:

            return (feature_names, get_feature(feature_names))
    except Exception as e:
        print (e)
        raise ValueError('Error in function ')
