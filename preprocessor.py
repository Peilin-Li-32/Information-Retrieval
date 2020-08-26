import nltk
nltk.download('wordnet')
from functools import lru_cache


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.

        self.stem = lru_cache(maxsize=10000)(nltk.PorterStemmer().stem)
        # self.stem = lru_cache(maxsize=10000)(nltk.stem.lancaster.LancasterStemmer().stem)
        # self.stem = lru_cache(maxsize=10000)(nltk.stem.snowball.SnowballStemmer("english").stem)
        # self.stem = lru_cache(maxsize=10000)(nltk.wordnet.WordNetLemmatizer().lemmatize)
        # self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize
        self.tokenize = nltk.tokenize.WordPunctTokenizer().tokenize

    def __call__(self, text):
        tokens = nltk.WordPunctTokenizer().tokenize(text)
        # tokens = nltk.WhitespaceTokenizer().tokenize(text)
        tokens = [self.stem(token) for token in tokens]
        return tokens

# bm25 + punct + Porter
# map: 0.470438997446486
# Rprec: 0.4866324535679375
# recip_rank: 0.631720430107527
# P_5: 0.4903225806451614
# P_10: 0.329032258064516
# P_15: 0.21935483870967745


# bm25 + white + Porter
# map: 0.4139633123532893
# Rprec: 0.4279639715123586
# recip_rank: 0.6413978494623656
# P_5: 0.4451612903225806
# P_10: 0.3
# P_15: 0.2


# bm25 + punct + snowball
# map: 0.4602568600754085
# Rprec: 0.4820241586370619
# recip_rank: 0.631720430107527
# P_5: 0.4838709677419356
# P_10: 0.32580645161290317
# P_15: 0.21720430107526884


# bm25 + white + snowball
# map: 0.4044779052872371
# Rprec: 0.42335567658148304
# recip_rank: 0.6198924731182796
# P_5: 0.4451612903225806
# P_10: 0.2967741935483871
# P_15: 0.1978494623655914


# bm25 + white + lancaster
# map: 0.39448485430340274
# Rprec: 0.39770283479960894
# recip_rank: 0.5986559139784946
# P_5: 0.4129032258064516
# P_10: 0.3
# P_15: 0.2


# bm25 + punct + lancaster
# map: 0.4455531076066791
# Rprec: 0.4634373690825304
# recip_rank: 0.6306835637480799
# P_5: 0.46451612903225814
# P_10: 0.32580645161290317
# P_15: 0.21720430107526884


# bm25 + punct + wordnet
# map: 0.2221236018845466
# Rprec: 0.26106339896662484
# recip_rank: 0.5922043010752689
# P_5: 0.2838709677419355
# P_10: 0.19354838709677413
# P_15: 0.12903225806451615


# bm25 + white + wordnet
# map: 0.2274353143016737
# Rprec: 0.2401340594888982
# recip_rank: 0.5857910906298004
# P_5: 0.3225806451612902
# P_10: 0.19032258064516122
# P_15: 0.12688172043010756


