import os
from features.phonetic import generate_phonetic_representation
from features.phonetic import get_stress_markers
from .loader import extract, pos_data, load_concepts, read_book
from nltk import sent_tokenize
import itertools
from helpers.decorators import lazy

# label extractors for book object

genre_le = lambda book: book.genre

success_le = lambda book: book.success

avg_rating_le = lambda book: book.avg_rating

genre_success_le = lambda book: "{}_{}".format(book.genre, book.success)


def success_3class_le(book):
    if book.avg_rating >= 3.5:
        return 2  # success
    elif book.avg_rating < 3.5 and book.avg_rating >= 2.5:
        return 1  # mild success
    else:
        return 0  # failure


class Book(object):
    """
    Wraps essential data for features extraction

    """

    def __init__(self, book_path, book_id, genre, success, avg_rating, sentic_file, stanford_parse_file):
        self.book_id = book_id
        self._book_path = book_path
        self.book_title = Book.book_title(self.book_id)
        self.genre = genre
        self.success = success
        self.avg_rating = avg_rating
        self.sentic_file = sentic_file
        self.stanford_parse_file = stanford_parse_file

    @staticmethod
    def book_title(book_id):
        # TODO
        return book_id

    # def _get_parse_file_name(self, parse_path, ext='_st_parser.txt'):
    #     parse_fpath = os.path.join(parse_path, self.book_id.replace('.txt', ext))
    #     if os.path.exists(parse_fpath):
    #         return parse_fpath
    #     else:
    #         raise OSError('Parsed file not found')
    #
    # def get_pos_path(self):
    #     # todo for emnlp
    #     return self.stanford_parse_file
    #     # return os.path.abspath(os.path.join(self._book_path, '../../st_parser'))
    #
    # def get_sentic_path(self):
    #     # todo for emnlp
    #     return self.sentic_file
    #     # return os.path.abspath(os.path.join(self._book_path, '../../sentic'))

    def read_st_parse_file(self):
        # pos_path = self.get_pos_path()
        # print(pos_path)
        # st_parse_file_name = self._get_parse_file_name(pos_path)
        st_parse_file_name = self.stanford_parse_file
        # self._word_pos, self._parse_tree, dependency = extract(os.path.join(pos_path, st_parse_file_name))
        self._word_pos, self._parse_tree, dependency = extract(st_parse_file_name)
        self._pos_tag = pos_data(self._word_pos)
        if hasattr(self, 'size'):
            self._pos_tag = '\n'.join(self._pos_tag.split('\n')[:self.size])
            self._parse_tree = self._parse_tree[:self.size]

    def read_sentic_concepts(self):
        # sentic_path = self.get_sentic_path()
        # sentic_parse_file_name = self._get_parse_file_name(sentic_path, ext='_st_parser.txt.json')
        sentic_parse_file_name = self.sentic_file
        # self._concepts, self._concepts_ls, self._sensitivity, self._attention, self._pleasantness, self._aptitude, self._polarity = load_concepts(
        #     os.path.join(sentic_path, sentic_parse_file_name))
        self._concepts, self._concepts_ls, self._sensitivity, self._attention, self._pleasantness, self._aptitude, self._polarity = load_concepts(
            sentic_parse_file_name)
        if hasattr(self, 'size'):
            self._sensitivity = self._sensitivity[:self.size]
            self._attention = self._attention[:self.size]
            self._pleasantness = self._pleasantness[:self.size]
            self._aptitude = self._aptitude[:self.size]
            self._polarity = self._polarity[:self.size]
            self._concepts = " ".join(list(itertools.chain.from_iterable(self._concepts_ls[:self.size])))

    def phonetic_representation(self):
        self._phonetics, self._phonetic_sent_words = generate_phonetic_representation(self.content)

    def stress_represenation(self):
        self._stress_markers = get_stress_markers(self.content, two_classes_only=True)

    def all_stress_represenation(self):
        self._all_stress_markers = get_stress_markers(self.content, two_classes_only=False)

    def of_size(self, size):
        content = ' '.join(sent_tokenize(self.content)[:size])
        sub_book = Book(book_id=self.book_id,
                        genre=self.genre, book_path=self._book_path, success=self.success, avg_rating=self.avg_rating,
                        sentic_file=self.sentic_file, stanford_parse_file=self.stanford_parse_file)
        sub_book.content = content
        setattr(sub_book, 'size', size)  # set the size attribute for the sub object

        return sub_book

    @property
    @lazy
    def content(self):
        self._content = read_book(self._book_path, encoding='latin1')
        return self._content

    @content.setter
    def content(self, content):
        self._content = content

    @property
    @lazy
    def concepts(self):
        self.read_sentic_concepts()
        return self._concepts

    @property
    @lazy
    def sensitivity(self):
        self.read_sentic_concepts()
        return self._sensitivity

    @property
    @lazy
    def attention(self):
        self.read_sentic_concepts()
        return self._attention

    @property
    @lazy
    def aptitude(self):
        self.read_sentic_concepts()
        return self._aptitude

    @property
    @lazy
    def polarity(self):
        self.read_sentic_concepts()
        return self._polarity

    @property
    @lazy
    def pleasantness(self):
        self.read_sentic_concepts()
        return self._pleasantness

    @property
    @lazy
    def pos_tag(self):
        self.read_st_parse_file()
        return self._pos_tag

    @property
    @lazy
    def parse_tree(self):
        self.read_st_parse_file()
        return self._parse_tree

    @property
    @lazy
    def phonetics(self):
        self.phonetic_representation()
        return self._phonetics

    @property
    @lazy
    def phonetic_sent_words(self):
        self.phonetic_representation()
        return self._phonetic_sent_words

    @property
    @lazy
    def word_pos(self):
        self.read_st_parse_file()
        return self._word_pos

    @property
    @lazy
    def stress_markers(self):
        self.stress_represenation()
        return self._stress_markers

    @property
    @lazy
    def all_stress_markers(self):
        self.all_stress_represenation()
        return self._all_stress_markers

    @property
    def book2vec_id(self):
        return '%s_%s' % (self.genre, self.book_id)
