
from sklearn.decomposition import TruncatedSVD,DictionaryLearning
from sklearn.preprocessing import Normalizer
import glob
import numpy
import csv
import os
import pathlib
from datetime import datetime
from sklearn import metrics
import numpy as np 

import argparse

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier,PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain,LabelPowerset
#from dataer import read_files_to_array
#from dataer import read_files_to_dict
from dataer import print_metrics_to_file
from dataer import create_result_dir
from plotter import plot_roc
from sklearn import metrics
from sklearn import datasets, linear_model
import os 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
lis=['/><br','<br','/>']
lis=[]
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix

from sklearn.neighbors import KNeighborsClassifier

def read_files_to_array(path):
    list = []
    files = glob.glob(path)
    for file in files:
        with open(file, encoding="utf-8") as f:
            list.append(f.read())
    return numpy.array(list)

SEED = 42
def extract():
    path1="train/neg/*.txt"
    path2="train/pos/*.txt"
    print('preparing data...')
    n_data_train = read_files_to_array(path1)
    p_data_train = read_files_to_array(path2)
    backup_n=n_data_train
    backup_p=p_data_train
        
    
    
    #n_data_train=backup_n
    #p_data_train=backup_p
    
    
    
    train_data = numpy.append(n_data_train, p_data_train)
    
    
    
    
    
    #pathtest="test/*.txt"
    #test_data = read_files_to_dict(pathtest)
    
    n_labels = [0] * len(n_data_train)
    p_labels = [1] * len(p_data_train)
    labels = n_labels + p_labels
    
    labelsy=np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(train_data, labelsy, test_size=0.2, random_state=SEED)
    print('done')
    
    
    ybackuptrain=np.copy(y_train)
    backupXtrain=np.copy(X_train)
    backupXval=np.copy(X_val)
    
    X_train=X_train.tolist()
    y_train=y_train.tolist()
    X_val=X_val.tolist()

    
    return X_train, X_val, y_train, y_val
