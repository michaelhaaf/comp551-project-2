1 Following is a brief description of the implementation: 

This implementation defines a Class for the BNB estimator. For utilization a naive object is first created. Then a fit() method is used to fit that particular object to an input X and an output Y (single dimension output and both in list format) 

Predict() method is then used to evaluate a prediction of the classification of an input X of any size. Predict method outputs the classification output Ypredicted. 

Ordinary Matrix operations wouldn't allow for an estimator that can handle large sparse matrices. Accordingly this estimator was made to handle the output of a vectorizer.fit_transform() method for generalizability. That is done by exploiting the sparse nature of the input and performing computation only for values registered as non zeros in the output of the vectorizer.fit_transform() method

The same dictionary (i.e. feature list) that fitting to X_train, Y_train is used in mapping a validation or test input X_val,X_train into an input matrix. 
 

2 Following is a brief description of the Test run and results obtained: 

for the parameters in these files an accuracy of 87.74 was obtained 

Naive Bayes Run Uses CountVectorizer to create an input and the creates a NB classifier object 
that object is the fitted into the training data which is directly followed by a prediction 


for the 80:20 split Xtrain is a 20,000 x 500,000 sparse matrix 
number of non zero elements is above 6 million 
sparse matrix operations are used to allow for the manipulation of matrices of this size 
be mindful that it might take from 5-10 minutes to run completely. 


CountVectorizer(max_features=500000, min_df=1,ngram_range=(1, 2), binary=True) with the following parameters was used 




3 Following are some information necessary to effectively run the code: 


Before Running the code you need to have you train test split ready. 
The X and Y values are converted to list for convenience (from numpy). 

Running 11/ data extract defines a function which extracts the data in suitable format if its in the directory. 



Running Naive_Bayes_Run.py after have ran (11/data extract.py and bernoulliN_Estimator_Class.py) will produce the results mentioned in the reports 