from collections import Counter
import nltk
import string
from nltk.corpus import cmudict
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from nltk.util import ngrams as nltk_ngrams


dirname = os.path.dirname(__file__)

__all__ = ['PhonemeGroupBasedFeatures', 'PhoneticCharNgramsVectorizer', 'generate_phonetic_representation','get_stress_markers','StressFeatures','StressNgramsVectorizer']

arpabet = cmudict.dict()

with open(os.path.join(dirname, 'resources', 'cmudict_phoneme_groups.json'), 'r') as f_in:
    word_phoneme_group = json.load(f_in)


class Trie():
    # main Trie, root node: unifier of individual root nodes of tries starting from different phoneme groups,
    # timesTraversed on root remains 0
    def __init__(self):
        self.times_traversed = 0
        self.children = {}  # Trie{a: Trie{b: Trie{...}}, b: Trie{...},...}

    # adds one phoneme group of a word at a time recursively
    def add_word(self, word, pop_order=0):
        if not word:
            return
        phoneme_group = word.pop(pop_order)
        if phoneme_group not in self.children:
            self.children[phoneme_group] = Trie()
        child = self.children[phoneme_group]
        child.times_traversed += 1
        child.add_word(word, pop_order)

    def get_counts(self):
        counts = []
        for child in self.children.values():
            counts.append(child.times_traversed)
            counts.extend(child.get_counts())
        return counts

    def __str__(self):
        result = "(" + str(self.times_traversed)
        for phoneme_group, child in self.children.iteritems():
            result += " " + str(phoneme_group)
            result += " " + str(child)
        result += ")"
        return result


def get_phonetic_representation(word):
    try:
        return arpabet[word][0]
    except:
        return None


def get_words_in_text(text, stopwords=False):  # get words per sentence
    if stopwords:
        stops = set(stopwords.words("english"))
        words = [word for word in nltk.word_tokenize(text.replace('\r\n', '')) if
                 word.lower() not in stops]
    else:
        words = [word for word in nltk.word_tokenize(text.replace('\r\n', ' '))]
    return words




def get_stress_markers(content, two_classes_only=True):
    content=content.lower()
    stress_phrase_sents = []
    for sent in nltk.sent_tokenize(content.replace('\r\n', ' ')):
        stress_sents = []
        stress_phrase = ""
        for word in get_words_in_text(sent):
            if word in string.punctuation:
                stress_sents.append(stress_phrase)
                stress_phrase = ""
            else:
                word = word.lower()
                phones = get_phonetic_representation(word)
                if phones:
                    for phoneme in phones:
                        if phoneme[-1] in ["0", "1"]:
                            stress_phrase += phoneme[-1]
                        elif phoneme[-1] == "2":
                            stress_phrase += "1" if two_classes_only else "2"
        if stress_sents:
            stress_phrase_sents.append(stress_sents)
    return stress_phrase_sents






def generate_phonetic_representation(content):
    phonetics = []
    phonetic_sent_words = []
    for sent in nltk.sent_tokenize(content.replace('\r\n', ' ')):
        phonetic_sent = []
        for word in get_words_in_text(sent):
            if word in string.punctuation:
                if len(phonetics) > 0:
                    phonetics.pop()  # remove the space
                phonetics.append(word)
            else:
                word = word.lower()
                phones = get_phonetic_representation(word)
                if phones:
                    phonetics.extend(phones + [' '])
                    phonetic_sent.append(word_phoneme_group[word])
        if phonetic_sent:
            phonetic_sent_words.append(phonetic_sent)
    if phonetics and phonetics[
        -1] == ' ':  # some tweets have words but none of the words are in the phonetic dictionary
        phonetics.pop()
    return phonetics, phonetic_sent_words


class PhoneticCharNgramsVectorizer(TfidfVectorizer):
    def _phonetic_char_ngrams(self, text_document):
        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams.append("".join(text_document[i: i + n]))
        return ngrams

    def build_analyzer(self):
        self.lowercase=False
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._phonetic_char_ngrams(preprocess(self.decode(doc.phonetics)))


class PhonemeGroupBasedFeatures(BaseEstimator,TransformerMixin):
    def get_feature_names(self):
        return np.array(["plosives", "fricatives", "homogeneity", "alliteration", "rhyme"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        plosive_groups = ['pb', 'td', 'kg']
        fricative_groups = ['fv', 'tdh', 'sz', 'szh', '18']
        plosives = []
        fricatives = []
        alliteration = []
        rhyme = []
        homogeneity = []
        for doc in documents:

            alliteration_score_acc = 0
            rhythm_score_acc = 0
            homogeneity_acc = 0
            for phonetic_sent in doc.phonetic_sent_words:
                # print ("phonetic sent", phonetic_sent)
                all_phonemes = []
                alliteration_trie = Trie()
                rhythm_trie = Trie()
                for phonetic_word in phonetic_sent:
                    # print ("phonetic word", phonetic_word)
                    all_phonemes.extend(phonetic_word)
                    alliteration_trie.add_word(phonetic_word[:])
                    rhythm_trie.add_word(phonetic_word[:], pop_order=-1)
                alliteration_counts = alliteration_trie.get_counts()
                alliteration_score_acc += sum([x for x in alliteration_counts if x > 1]) / sum(alliteration_counts)
                rhythm_counts = rhythm_trie.get_counts()
                rhythm_score_acc += sum([x for x in rhythm_counts if x > 1]) / sum(rhythm_counts)
                # print ("scores", alliteration_counts, rhythm_counts)
                all_phonemes_ctr = Counter(all_phonemes)
                homogeneity_acc += 1 - len(all_phonemes_ctr) / len(all_phonemes)
            try:
                plosives.append(sum([all_phonemes_ctr[x] for x in plosive_groups]) / len(all_phonemes))
            except ZeroDivisionError as ex:
                 plosives.append(0)

            try:
                fricatives.append(sum([all_phonemes_ctr[x] for x in fricative_groups]) / len(all_phonemes))
            except ZeroDivisionError as ex:
                 fricatives.append(0)
            try:
                homogeneity.append(homogeneity_acc / len(doc.phonetic_sent_words))
            except ZeroDivisionError as ex:
                 homogeneity.append(0)
            try:
                alliteration.append(alliteration_score_acc / len(doc.phonetic_sent_words))
            except ZeroDivisionError as ex:
                 alliteration.append(0)
            try:
                rhyme.append(rhythm_score_acc / len(doc.phonetic_sent_words))
            except ZeroDivisionError as ex:
                rhyme.append(0)

        X = np.array([plosives, fricatives, homogeneity, alliteration, rhyme]).T
        return X





class StressFeatures(BaseEstimator,TransformerMixin):
    def get_feature_names(self):
        return np.array(["repeated", "mirror", "repeated_phrase"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        repeated = []
        mirror = []
        repeated_phrase = []
        for doc in documents:
            repeated_count = 0
            mirror_count = 0
            repeated_phrase_count = 0
            stress_phrase_sents = doc.stress_markers
            # print(stress_phrase_sents)
            for i in range(0, len(stress_phrase_sents)):
                if i==1 and stress_phrase_sents[i] == stress_phrase_sents[i-1] and len(stress_phrase_sents[i]) == 1:
                    repeated_count += 1
                else:
                    phrases_in_sent = stress_phrase_sents[i]
                    # print(phrases_in_sent)
                    phrase_combinations = list(nltk_ngrams(phrases_in_sent, 2)) + list(nltk_ngrams(phrases_in_sent, 3))
                    for pc in phrase_combinations:
                        str = "".join(pc)
                        if str == str[::-1]:
                            # print("mirror", str)
                            mirror_count += 1
                        str_length = len(str)
                        first_half = str[:str_length//2]
                        second_half = str[str_length//2:] if str_length % 2 == 0 else str[str_length//2+1:]
                        if first_half == second_half:
                            # print("rp", str)
                            repeated_phrase_count += 1
            repeated.append(repeated_count)
            mirror.append(mirror_count)
            repeated_phrase.append(repeated_phrase_count)
        X = np.array([repeated, mirror, repeated_phrase]).T
        return X



class StressNgramsVectorizer(TfidfVectorizer):

    def _stress_ngrams(self, stress_phrase_sents):
        ngrams = []
        min_n, max_n = self.ngram_range
        for sent in stress_phrase_sents:
            stress_sent = "".join(sent)
            for n in range(min_n, min(max_n + 1, len(stress_sent) + 1)):
                for i in range(len(stress_sent) - n + 1):
                    ngrams.append("".join(stress_sent[i: i + n]))
            # print(sent)
            # print("ngrams",ngrams)
        return ngrams

    def build_analyzer(self):
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._stress_ngrams(doc.all_stress_markers)



if __name__ == '__main__':
    # repetition  of similar stress patterns either within a phrase or  between consecutive phrases
    #single phrase identical in syntax, syllable structuer and stress
    #phrase markers: ,
    #same stress , syllable structure and syntax
    # paired parallel phrases ( how fresh, how calm,  -> like the flap of a wave, the kiss of a wave)

    text="""What a lark! What a plunge! For so it had always seemed to her, when, with a little squeak of the hinges,
    which she could hear now, she had burst open the French windows and plunged at Bourton into the open air.
    How fresh, how clam, stiller than this of course, the air was in the early morning; like the flap of a wave, the kiss of a wave, chill and sharp and yet
    (for a girl of eighteen as she then was) solemn, feeling as she did,  standing there at the open window, that something awful was about to happen.
    """

    # print (get_stress_markers(text))
    #
    # book = BookDataWrapper(book_id="b1", content=text)
    # sf = StressFeatures()
    # print(sf.fit_transform([book]))
    #
    # sngrams = StressNgramsVectorizer(ngram_range=(3, 3))
    # print (sngrams.fit_transform([book]))


