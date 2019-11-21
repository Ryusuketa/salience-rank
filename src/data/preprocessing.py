import numpy as np
import networkx as nx
import itertools
from typing import Dict, Set, List, Union, Tuple
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary


def get_terms_included_in_corpus(corpus: List[List[str]]) -> List[str]:
    return list(set(itertools.chain.from_iterable(corpus)))


class WordProbabilityModel(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary
        self._term_probability: Dict[int, float] = dict()

    def fit(self, corpus: List[List[str]]) -> float:
        terms = get_terms_included_in_corpus(corpus)
        token2id = self._dictionary.token2id
        cfs = self._dictionary.cfs
        self._term_probability = {token2id[term]: cfs[token2id[term]] / self._dictionary.num_pos for term in terms}

    def predict(self, term: Union[str, List[str]]) -> float:
        if type(term) == str:
            return self._term_probability.get(term)
        elif type(term) == list:
            return [self._term_probability.get(t) for t in term]
        else:
            raise TypeError("Input valiable type error. `str` or `list` must be input.")


class TopicSpecifityModel(object):
    def __init__(self, lda_model: LdaModel, dictionary: Dictionary) -> None:
        self._lda_model = lda_model
        self._dictionary = dictionary
        self._max_specifity = np.nan
        self._topic_probability = dict()
        self._term_probability = dict()
        self._topic_specifity = dict()

    def fit(self, corpus: List[List[str]]):
        terms = get_terms_included_in_corpus(corpus)
        self._topic_probability = self._calculate_topic_probability(terms)
        self._topic_specifity = dict(zip(range(len(terms)),
                                         list(map(lambda index: self._calculate_topic_specifity_given_word(terms[index]))), range(len(terms))))

    def predict(self, term: Union[str, List[str]]) -> float:
        if type(term) == str:
            return self._term_probability.get(term)
        elif type(term) == list:
            return [self._term_probability.get(t) for t in term]
        else:
            raise TypeError("Input valiable type error. `str` or `list` must be input.")

    def _calculate_topic_specifity_given_word(self, term: str) -> float:
        topic_prob_given_term = dict(self._lda_model.get_term_topics(term))
        topic_prob_given_term = [topic_prob_given_term[i] for i in range(len(topic_prob_given_term))]
        topic_prob = [self._topic_probability[i] for i in range(len(self._topic_probability))]

        return self.calculate_kl_divergence(topic_prob_given_term, topic_prob)

    @staticmethod
    def calculate_kl_divergence(topic_prob_given_term: List[float], topic_prob: List[float]) -> float:
        return sum(map(lambda t_w, t: t_w * np.log(t_w / t), topic_prob_given_term, topic_prob))

    def _calculate_topic_probability(self, terms: List[str]) -> Dict[int, float]:
        def calculate_merginal_probability(topic_probabilities_given_terms: Dict[int, [List[Tuple[int, float]]]],
                                           term_probabilty: Dict[int, float],
                                           topic_id: int) -> float:
            return sum([topic_probabilities_given_terms[term_id][topic_id][1] * term_probabilty[term_id] for term_id in range(len(term_probabilty))])

        topic_probabilities_given_terms = map(lambda term_id: self._lda_model.get_term_topics(term_id), range(len(self._term_probability)))
        topic_probabilities = map(lambda topic_id: calculate_merginal_probability(topic_probabilities_given_terms, self._term_probability, topic_id),
                                  range(len(self._lda_model.n_topic)))
        return dict(zip(range(self._lda_model.n_topic), topic_probabilities))


class SalienceRankGraph(object):
    def __init__(self,
                 window_size: int, 
                 salience_model: TopicSpecifityModel,
                 corpus_unigram: WordProbabilityModel,
                 topic_specifity_model: TopicSpecifityModel,
                 salience_balance: float,
                 salience_weight: float,
                 topic_corpus_propotion: float):
        self.G = nx.Graph()
        self._window_size = window_size
        self._corpus_unigram = corpus_unigram
        self._topic_specifity_model = topic_specifity_model
        self._topic_corpus_propotion = topic_corpus_propotion
        self._salience_weight = salience_weight

    def _add_token_to_cooccurrence_graph(self, G: nx.Graph, corpus: List[List[str]]) -> None:
        for tokens in corpus:
            pairs = self._get_cooccurred_token_pairs(tokens)
            G.add_edge_from(pairs)

    def _get_cooccurred_token_pairs(self, tokens: List[str]) -> List[Tuple[str, str]]:
        pairs = [(tokens[i + j], tokens[i]) for i in range(1, self._window_size) for j in range(len(tokens) - i)]
        return pairs

    def _calculate_topic_salience(self, term: str):
        return self._topic_specifity_model.predict(term)

    def _calculate_corpus_salience(self, term: str):
        return self._corpus_unigram.predict(term)

    def _calculate_term_salience(self, corpus: List[List[str]]):
        def calculate_salience(term: str) -> float:
            alpha = self._topic_corpus_propotion
            return (1 - alpha) * self._calculate_corpus_salience(term) + alpha * self._calculate_topic_salience(term)
        terms = get_terms_included_in_corpus(corpus)
        return dict(zip(terms, map(lambda term: calculate_salience(term), terms)))

    def _rank(self, G: nx.Graph, term_salience: Dict[str, float]) -> Dict[str, float]:
        return nx.pagerank(G, personalized=term_salience, alpha=self._salience_weight)

    def predict(self, corpus: List[List[str]]):
        self._add_token_to_cooccurrence_graph(self.G, corpus)
        scores = self._rank(self.G, self._calculate_term_salience(corpus))
        return scores
