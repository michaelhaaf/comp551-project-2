from sklearn.base import BaseEstimator, TransformerMixin

class CustomNB(BaseEstimator, TransformerMixin):


    def transform(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

