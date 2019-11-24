import unittest
from gensim.corpora import Dictionary
from src.model import WordProbabilityModel


class TestWordProbabilityModel(unittest.TestCase):
    def setUp(self):
        texts = [['a', 'b', 'c'], ['b', 'a', 'c', 'd']]
        dictionary = Dictionary(texts)
        self.model = WordProbabilityModel(dictionary)

    def test_fit(self):
        corpus = [['a', 'b', 'c'], ['d', 'e', 'c']]
        self.model.fit(corpus)
        expects = {0: 2/7, 1: 2/7, 2: 2/7, 3: 1/7}
        results = self.model._term_probability
        self.assertEqual(expects, results)
