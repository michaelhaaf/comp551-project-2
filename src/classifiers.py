from sklearn.base import BaseEstimator, TransformerMixin

class CustomNB(BaseEstimator, TransformerMixin):
    def apply(self, tags):
        return self

    def transform(self, tags_list, y=None):
        return self

    def fit(self, tags_list, y=None):
        return self

