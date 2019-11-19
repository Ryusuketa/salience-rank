from nltk import SpaceTokenier
import numpy as np
import networkx as nx
import itertools
from typing import Dict, Set
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary


def get_terms_included_in_corpus(corpus: List[List[str]]) -> Set[str]:
    return set(itertools.chain.from_iterable(corpus))

class WordProbabilityModel(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary
        self._term_probability: Dict[int, float] = dict()

    def fit(self, corpus: List[List[str]]) -> float:
        terms = get_terms_included_in_corpus(corpus)
        token2id = self._dictionary.token2id
        cfs = self._dictionary.cfs
        self._term_probability = {token2id[term]: cfs[token2id[term]] / self._dictionary.num_pos}

    def predict(self, term: str) -> float:
        pass

class TopicSpecifityModel(object):
    def __init__(self, lda_model: LdaModel, dictionary: Dictionary) -> None:
        self._lda_model = lda_model
        self._dictionary = dictionary
        self._max_specifity = None
        self._topic_probability = None
        self._term_probability = None

    def fit(self, corpus: List[List[str]]) -> TopicSpecifityModel:
        self.

    def predict(self, term: List[str]) -> float:
        
    def _calculate_topic_specifity(self, terms: List[str]) -> Dict[int, float]: 
        def calculate_merginal_distribution(topic_probabilities_given_terms: Dict[int, [List[Tuple[int, float]]]], term_probabilty: Dict[int, float], topic_id: int) -> float:
            return sum([topic_probabilities_given_terms[term_id][topic_id][1] * term_probabilty[term_id] for topic_probability in list_topics])

        topic_probabilities_given_terms = map(lambda term_id: self._lda_model.get_term_topics(term_id), term_ids)
        topic_probabilities = map(lambda topic_id: calculate_merginal_distribution(topic_probabilities_given_terms, range(self._lda_model.n_topic)))
        return dict(zip(range(self._lda_model.n_topic, topic_probabilities))) 
        

class SalienceRankGraph(object):
    def __init__(self, window_size: int, salience_model: TopicSpecifityModel, salience_balance: float, salience_weight: float, topic_corpus_propotion: float):
        self.G = nx.Graph()
        self._window_size = window_size
        self._corpus_unigram = corpus_unigram
        self._topic_specifity_model = _topic_specifity_model
        self._topic_corpus_propotion = topic_corpus_propotion
        self._salience_weight = salience_weight

    @staticmethod
    def add_token_to_cooccurrence_graph(G: nx.Graph, corpus: List[List[str]]) -> None:
        for tokens in corpus:
            pairs = self._get_cooccurred_token_pairs(tokens)
            G.add_edge_from(pairs)

    def _get_cooccurred_token_pairs(self, tokens: List[str]) -> List[Tuple[str, str]]:
        pairs = [(tokens[i + j], tokens[i]) for i in range(1, self._window_size) for j in range(len(tokens) - i))]
        return pairs

    def _calculate_topic_salience(self, term: str):
        return self._topic_specifity_model.predict(term)

    def _calculate_corpus_salience(self, term: str):
        return self._corpus_unigram.predict(term)

    def _calculate_term_salience(self, corpus: List[List[str]]):
        def calculate_salience(term: str) -> float:
            return (1 - self.alpha) * self._calculate_corpus_salience(term) + self.alpha * self._calculate_topic_salience(term)
        return dict(zip(terms, map(lambda term: calculate_salience(term), terms)))

    @staticmethod
    def _rank(G: nx.Graph, term_salience: Dict[str, float]) -> Dict[str, float]:
        return nx.pagerank(G, personalized=term_salience, alpha=self.salience_weight)
    
    def predict(self, corpus: List[List[str]]) -> self:
        self._add_token_to_cooccurrence_graph(self.G, c)
        scores = self._rank(self.G, self._calculate_term_salience(corpus))
        return scores
