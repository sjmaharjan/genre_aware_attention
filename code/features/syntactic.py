from collections import defaultdict
from nltk import Tree, string_types, Production, Nonterminal, ParentedTree
from nltk.tree import _child_names
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

__all__ = ['POSTags', 'Constituents', 'LexicalizedProduction', 'UnLexicalizedProduction',
           'GrandParentLexicalizedProduction', 'GrandParentUnLexicalizedProduction']


class POSTags(TfidfVectorizer):
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
            extract pos tags from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.pos_tag))


###################################################################################################


class Constituents(BaseEstimator):
    """Estimator to estimates the constituents features

    Analyze the parse tree and computes the normalized frequency for
    phrasal and clausal tags defined by methods:  phrasal_tags, clausal_tags

    Parameters
    ----------
    PHR : boolean {default: False}
        whether to compute the phrasal features

    CLS : boolean {default: False}
        whether to compute clausal features

    """

    def __init__(self, PHR=False, CLS=False):
        self.PHR = PHR
        self.CLS = CLS

    def get_feature_names(self):
        """ List of features names

        Parameters
        -----------
        self

        Returns
        --------
        feature_names : numpy array list of features names


        """

        if self.PHR and self.CLS:
            return np.array(self.phrasal_tags + self.clausal_tags)
        elif self.PHR:
            return np.array(self.phrasal_tags)
        elif self.CLS:
            return np.array(self.clausal_tags)
        else:
            return np.array([])

    def _get_phrase_count(self, parse_trees):
        """Helper method to count the  phrasal consituent in the tree

        """
        return self._count_constituents(self.phrasal_tags, parse_trees)

    def _get_clause_count(self, parse_trees):
        """Helper method to count the  clausal consituent in the tree

        """
        return self._count_constituents(self.clausal_tags, parse_trees)

    def _count_constituents(self, tags, parse_trees):
        """Helper method to count the provide set of tags in the tree

        """

        d = defaultdict(int)
        for tree in parse_trees:
            for phrase in tags:
                d[phrase] += tree.count('(' + phrase + ' ')  # count '(S '
        lst = []
        for phrase in tags:
            lst.append(d[phrase])
        return lst

    def fit(self, documents, y=None):
        """Fit the training data
        Estimator does not have to learn anything so just return self

        Returns
        --------
        self

        """
        return self

    def transform(self, documents):
        """ Transform the documents into a features matrix
            with columns as features names and rows as document

        Parameters
        -----------
        documents : list, iterable

        Returns
        ----------
        features matrix : numpy array, shape [number of documents, number of features]


        """
        result = []
        for d in documents:
            r1, r2 = [], []
            parse_tree = d.parse_tree
            # remove \n in parse tree in string representation
            parse_tree = [tree.replace('\n', ' ') for tree in parse_tree]

            if self.PHR:
                phrase_count = self._get_phrase_count(parse_tree)
                total_phrase_count = sum(phrase_count)
                if total_phrase_count == 0:
                    r1 = [0.0 for x in phrase_count]
                else:
                    r1 = [x * 1.0 / total_phrase_count for x in phrase_count]

            if self.CLS:
                clause_count = self._get_clause_count(parse_tree)
                total_clause_count = sum(clause_count)
                if total_clause_count == 0:
                    r2 = [0.0 for x in clause_count]
                else:
                    r2 = [x * 1.0 / total_clause_count for x in clause_count]

            result.append(r1 + r2)

        # print result
        return np.array(result)

    def fit_transform(self, documents):
        return self.transform(documents)

    @property
    def phrasal_tags(self):
        return ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP',
                'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']

    @property
    def clausal_tags(self):
        return ['SBAR', 'SQ', 'SBARQ', 'SINV', 'S']


###################################################################################################

# Grammer Rules

class ParseTree(Tree):
    """ Extends the nltk.Tree class

    Adds methods to return the lexicalized, unlexicalized with or w/o grand-parent nodes


    """

    def lexicalized_productions(self):
        """
        Returns
        ---------
        Preoduction rules with lexicons (terminals/words)

        """
        return self.productions()

    def unlexicalized_productions(self):
        """
        Returns
        ---------
        Preoduction rules with out lexicons (terminals/words)

        """
        return filter(lambda x: x.is_nonlexical(), self.productions())

    def grandparent_lexicalized_productions(self):
        """
        Returns
        ---------
        Preoduction rules with lexicons (terminals/words) and grand-parent node (node above the considers pr)

        """

        ptree = self.parent_tree
        return self._gp_lexicalized_rules(ptree)

    def grandparent_unlexicalized_productions(self):
        """
        Returns
        ---------
        Preoduction rules with out lexicons (terminals/words) and grand-parent node (node above the considers pr)

        """

        return filter(lambda x: x.is_nonlexical(), self.grandparent_lexicalized_productions())

    def _gp_lexicalized_rules(self, ptree):
        """Helper method to extract grandparent nodes and the production rules
            from the tree

        Parameters
        ------------
        ptree : nltk Tree object

        Returns
        ---------
        production_rules : list
            add grand-parent nodes to prodcution ruels with '^'

        """

        if not isinstance(ptree._label, string_types):
            raise TypeError('Productions can only be generated from trees having node labels that are strings')
        prods = []
        for child in ptree.subtrees():
            if child.parent():
                prods += [Production(Nonterminal(child.parent().label() + '^' + child._label), _child_names(child))]

        return prods

    @property
    def parent_tree(self):
        return ParentedTree.convert(self)


###################################################################################################

class LexicalizedProduction(TfidfVectorizer):
    """Convert a collection of  parse tree of documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
    """

    def _productions(self, parse_trees):
        productions = []
        for tree in parse_trees:
            productions += ParseTree.fromstring(tree).lexicalized_productions()
        return productions

    def build_analyzer(self):
        return lambda doc: (w for w in self._productions(doc.parse_tree))


###################################################################################################

class UnLexicalizedProduction(TfidfVectorizer):
    def _productions(self, parse_trees):
        productions = []
        for tree in parse_trees:
            productions += ParseTree.fromstring(tree).unlexicalized_productions()
        return productions

    def build_analyzer(self):
        return lambda doc: (w for w in self._productions(doc.parse_tree))


###################################################################################################

class GrandParentLexicalizedProduction(TfidfVectorizer):
    def _productions(self, parse_trees):
        productions = []
        for tree in parse_trees:
            productions += ParseTree.fromstring(tree).grandparent_lexicalized_productions()
        return productions

    def build_analyzer(self):
        return lambda doc: (w for w in self._productions(doc.parse_tree))


###################################################################################################

class GrandParentUnLexicalizedProduction(TfidfVectorizer):
    def _productions(self, parse_trees):
        productions = []
        for tree in parse_trees:
            productions += ParseTree.fromstring(tree).grandparent_unlexicalized_productions()
        return productions

    def build_analyzer(self):
        return lambda doc: (w for w in self._productions(doc.parse_tree))


if __name__ == '__main__':
    from booxby.models import  Book
    from booxby import create_app
    import os
    from dump_vectors import BookDataWrapper

    app=create_app(os.getenv('FLASK_CONFIG') or 'default')
    for book in Book.objects(is_active=True):
        # print (book.content)
        b=BookDataWrapper(book_id=book.book_id,isbn_10=book.isbn_10,content='')
        print(b.parse_tree[0])
        print(b.parse_tree[-1])
        print (len(b.parse_tree))

        break