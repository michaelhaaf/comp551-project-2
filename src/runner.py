#!/usr/bin/env python

import argparse

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.dataer import *
from src.dataer import print_metrics_to_file
from src.plotter import plot_roc

SEED = 42

classifier_dict = {
    'mnb': MultinomialNB(),
    'bnb': BernoulliNB(),
    'log_reg': LogisticRegression(),
    'lin_svm': LinearSVC(),
    'sgd': SGDClassifier()
}

feature_dict = {
    'both_gram': CountVectorizer(ngram_range=(1,2))
}

cv_parameters = {
    'clf__alpha': (0, 0.5, 1.0)
}

def construct_pipeline(selected_features, selected_classifier):
    feature_pipelines = construct_feature_pipelines(selected_features)
    return Pipeline([
        ('features', FeatureUnion(feature_pipelines)),
        ('clf', classifier_dict[selected_classifier])
    ])


def construct_feature_pipelines(selected_features):
    return [(f, make_pipeline(feature_dict[f])) for f in selected_features]


# def get_feature_names_from_pipeline(pipeline):
#     return list(pipeline.named_steps['features'].transformer_list[0][1].named_steps.values())[1].get_feature_names()


def main(cmd_args):

    ### prepare data
    print('preparing data...')
    n_data_train = read_files_to_array('../data/train/neg/*.txt')
    p_data_train = read_files_to_array('../data/train/pos/*.txt')
    train_data = numpy.append(n_data_train, p_data_train)

    test_data = read_files_to_dict('../data/test/*.txt')

    n_labels = [0] * len(n_data_train)
    p_labels = [1] * len(p_data_train)
    labels = n_labels + p_labels

    X_train, X_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.20, random_state=SEED)
    print('done')

    ### prepare feature/classifier pipeline
    print('preparing feature/classifier pipeline...')
    pipeline = construct_pipeline(cmd_args.selected_features, cmd_args.selected_classifier)
    grid_search = GridSearchCV(pipeline, cv_parameters, cv=5, n_jobs=-1, verbose=100)

    classifier = None
    if cmd_args.perform_cv:
        classifier = grid_search
    else:
        classifier = pipeline

    classifier.fit(X_train, y_train)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='attempt to classify the authors of some parsed texts')
    parser.add_argument('--selected_features', nargs='+', choices=feature_dict.keys(), default='both_gram',
                        help='the feature sets to analyze. selecting more than one will combine all features in parallel')
    parser.add_argument('--selected_classifier', choices=classifier_dict.keys(), default='nb',
                        help='the model to use')
    parser.add_argument('--perform_cv', action='store_true',
                        help='if selected, perform cross-validation. recommended for final results, not for testing')
    args = parser.parse_args()

    main(args)