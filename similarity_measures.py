from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt


class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """

    def __init__(self, postings):
        # If do not put dictionary token_to_idf in the class,
        # it need to be calculated every time, which is not efficient.
        # So, put this dictionary in the class part and only calculate it for once.
        self.token_to_idf_TFIDF = defaultdict(lambda: 0)
        self.token_to_idf_BM25 = defaultdict(lambda: 0)
        self.doc_to_len_BM25 = defaultdict(lambda: 0)
        self.doc_to_token_BM25 = defaultdict(dict)
        self.avgdl = 0
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]


class TFIDF_Similarity(CosineSimilarity):
    # TODO implement the set_document_norms and get_scores methods.
    # Get rid of the NotImplementedErrors when you are done.
    def set_document_norms(self):
        # IDF for every term is same, make a dic to keep the IDFs
        for token, doc_to_counts in self.postings.token_to_doc_counts.items():
            df = 0
            for count in self.postings.token_to_doc_counts[token].values():
                if count > 0:
                    df += 1
            N = len(self.postings.doc_to_token_counts.keys())
            self.token_to_idf_TFIDF[token] += log((N / df))

        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(
                sum([(token_counts[token] * self.token_to_idf_TFIDF[token]) ** 2 for token in token_counts.keys()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc in self.postings.token_to_doc_counts[token].keys():
                tf_idf = self.postings.doc_to_token_counts[doc][token] * self.token_to_idf_TFIDF[token]
                doc_to_score[doc] += query_term_frequency * self.token_to_idf_TFIDF[token] * tf_idf / self.doc_to_norm[doc]


class BM25_Similarity(CosineSimilarity):

    def set_document_norms(self):
        # IDF for every term is same, make a dic to keep the IDFs
        for token, doc_to_counts in self.postings.token_to_doc_counts.items():
            df = 0
            for count in self.postings.token_to_doc_counts[token].values():
                if count > 0:
                    df += 1
            N = len(self.postings.doc_to_token_counts.keys())
            self.token_to_idf_BM25[token] += log(((N-df+0.5) / (df+0.5)), 10)

        all_len = 0
        # make a dic to keep the lens of every doc
        for doc, token_to_counts in self.postings.doc_to_token_counts.items():
            this_len = 0
            for count in token_to_counts.values():
                if count > 0:
                    this_len += count
                    all_len += count
            self.doc_to_len_BM25[doc] = this_len

        # keep the avgdl
        self.avgdl = all_len/len(self.postings.doc_to_token_counts.keys())

        for doc, token_counts in self.postings.doc_to_token_counts.items():
            s = 0
            d = {}
            for token, count in token_counts.items():
                idf_q = self.token_to_idf_BM25[token]
                tf_qd = count
                k1 = 2.0
                b = 0.75
                d_len = self.doc_to_len_BM25[doc]
                score_qd = idf_q * tf_qd * (k1+1) / (tf_qd + k1*(1-b+b*d_len/self.avgdl))
                d[token] = score_qd
                s = s + d[token] ** 2
            self.doc_to_token_BM25[doc] = d
            self.doc_to_norm[doc] = sqrt(s)

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc in self.postings.token_to_doc_counts[token].keys():
                doc_to_score[doc] += self.doc_to_token_BM25[doc][token]