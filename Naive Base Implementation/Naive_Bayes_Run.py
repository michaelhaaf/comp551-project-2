
#X_train, X_val, y_train, y_val = extract()
#run the split function 
from math import log
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import numpy 
from scipy.sparse import find

X_train, X_val, y_train, y_val = extract()

#X_train=X_train.tolist()
#y_train=y_train.tolist()
#X_val=X_val.tolist()
X=X_train
Y=y_train





corpus=X
vectorizer1=CountVectorizer(max_features=500000, min_df=1,ngram_range=(1, 2), binary=True)
input_vector=vectorizer1.fit_transform(corpus)    # a sparse matrix 
words=vectorizer1.get_feature_names()  # feature list which is used to compute the input vector for X_val 
vectorizer2=CountVectorizer(max_features=500000, min_df=1,ngram_range=(1, 2), binary=True,vocabulary=words)
X_val=vectorizer2.fit_transform(X_val) 



estimator=bernoulliNB_Estimator()  # defines an estimator bernoulliNB object 
estimator.fit(input_vector,Y,words) # fits estimator object to data  
y_val_results_BNB = estimator.predict(X_val) # classifies validation set 
print(metrics.accuracy_score(y_val, y_val_results_BNB)) # accuracy measure 87.74 
