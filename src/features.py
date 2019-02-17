import numpy

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

class SentimentScorer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform(self, X, y=None):
        scores = []
        for comment in X:
            sentences = nltk.sent_tokenize(comment)
            sent_scores = []
            for s in sentences:
                sent_scores.append(self.sia.polarity_scores(s)['compound'])
            scores.append(numpy.mean(sent_scores))
        return numpy.array(scores).reshape(-1, 1)

    def fit(self, X, y=None):
        return self








