import glob
import numpy
import csv
import os
import pathlib
from datetime import datetime

from sklearn import metrics


def read_files_to_array(path):
    list = []
    files = glob.glob(path)
    for file in files:
        with open(file) as f:
            list.append(f.read())
    return numpy.array(list)


def read_files_to_dict(path):
    dict = {}
    files = glob.glob(path)
    for file in files:
        with open(file) as f:
            base = os.path.basename(file)
            id = os.path.splitext(base)[0]
            dict[id] = f.read()
    return dict


def create_submission_file(res_dir, keys, y_results):
    with open(res_dir + '/submission.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for key, result in zip(keys, y_results):
            csv_writer.writerow([key, result])


def create_result_dir(cmd_args):
    now_str = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    dir_name = '../results/' \
               + str(cmd_args.selected_features[0]) + '_' \
               + str(cmd_args.selected_classifier) + '_' \
               + 'cv=' + str(cmd_args.perform_cv) + '_' \
               + now_str
    pathlib.Path(dir_name).mkdir(exist_ok=True)
    return dir_name


def print_metrics_to_file(res_dir, classifier, cmd_args, y_val, y_val_results):
    with open(res_dir + '/metrics.txt', 'w') as f:
        if cmd_args.perform_cv:
            print(classifier.best_params_, file=f)
            print('overall accuracy: ', classifier.best_score_, file=f)
        else:
            print('overall accuracy: ', metrics.accuracy_score(y_val, y_val_results), file=f)
        print(metrics.classification_report(y_val, y_val_results), file=f)