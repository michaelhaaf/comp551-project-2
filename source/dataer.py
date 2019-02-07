import glob
import numpy
import csv
import os

from datetime import datetime


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


def print_results_to_csv(keys, y_results, cmd_args):
    now_str = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    filename = '../results/' + now_str + '.csv'
    # to add: selected features/model from o cmd_args to filename

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Id', 'Category'])
        for key, result in zip(keys, y_results):
            csvwriter.writerow([key, result])
