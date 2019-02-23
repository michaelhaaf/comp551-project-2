#!/usr/bin/env python

import argparse
import multiprocessing
import timeit

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

from dataer import *
from dataer import print_metrics_to_file
from plotter import plot_roc
from goodreads_utilities import rand, random_subset

SEED = 42

classifier_dict = {
    'mnb': MultinomialNB(),
    'bnb': BernoulliNB(),
    'log_reg': LogisticRegression(),
    'lin_svm': LinearSVC(C=0.8),
    'sgd': SGDClassifier()
}

cv_parameters = {
    'clf__C': (0.8, 0.7)
}

stop_list=['read','reading','books','book','author',
           'write','writer','writers','writing','written'
           ,'wrote']
def construct_pipeline(selected_classifier):
    return Pipeline([
        ('ft', TfidfVectorizer(sublinear_tf=True,token_pattern= r'\w{1,}',stop_words=stop_list,max_df=0.55,
                               max_features=500000, min_df=1, ngram_range=(1, 2), binary=False)),
        ('clf', classifier_dict[selected_classifier])
    ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='attempt to classify the authors of some parsed texts')
    parser.add_argument('--selected_classifier', choices=classifier_dict.keys(), default='nb',
                        help='the model to use')
    parser.add_argument('--perform_cv', action='store_true',
                        help='if selected, perform cross-validation. recommended for final results, not for testing')
    cmd_args = parser.parse_args()

    ### prepare data
    print('preparing data...')
    n_data_train = read_files_to_array('../data/train/neg/*.txt')
    p_data_train = read_files_to_array('../data/train/pos/*.txt')
    train_data = numpy.append(n_data_train, p_data_train)

    test_data = read_files_to_dict('../data/test/*.txt')

    n_labels = [0] * len(n_data_train)
    p_labels = [1] * len(p_data_train)
    labels = n_labels + p_labels

    X_train, X_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.2, random_state=SEED)
    print('done')

    ### append Goodreads feature info
    X_train = X_train.tolist()
    X_val = X_val.tolist()

    size_of_RS = 8000
    RS_seed = 81821
    r = 10
    X_goodreads, y_goodreads = rand(r)
    X_goodreads, y_goodreads = random_subset(X_goodreads, y_goodreads, RS_seed, size_of_RS)
    X_train = X_train + X_goodreads
    y_train = y_train + y_goodreads

    ### prepare feature/classifier pipeline
    print('preparing feature/classifier pipeline...')
    pipeline = construct_pipeline(cmd_args.selected_classifier)
    grid_search = GridSearchCV(pipeline, cv_parameters, cv=5, n_jobs=4, verbose=100)

    classifier = None
    if cmd_args.perform_cv:
        classifier = grid_search
    else:
        classifier = pipeline

    start_time = timeit.default_timer()
    classifier.fit(X_train, y_train)
    print(timeit.default_timer() - start_time)
    print('done')


    ### predict
    print('performing prediction/evaluation...')
    y_val_results = classifier.predict(X_val)
    y_test_results = classifier.predict(test_data.values())


    ### print results
    print('generating reports...')
    result_dir = create_result_dir(cmd_args)
    create_submission_file(result_dir, test_data.keys(), y_test_results)
    print_metrics_to_file(result_dir, classifier, cmd_args, y_val, y_val_results)
    plot_roc(result_dir, y_val, y_val_results)


