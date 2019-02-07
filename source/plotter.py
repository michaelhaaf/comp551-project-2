import itertools

import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def confusion_matrix_best(X_train, X_test, y_train, y_test):
    naive_bayes_clf = MultinomialNB().fit(X_train, y_train)
    bayes_pred = naive_bayes_clf.predict(X_test)
    cm = confusion_matrix(y_test, bayes_pred)

    # code below borrowed from scikit-learn example
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix: naive bayes, bothgram, stop words kept')
    plt.colorbar()
    tick_marks = numpy.arange(2)
    plt.xticks(tick_marks, ['neg', 'pos'], rotation=45)
    plt.yticks(tick_marks, ['neg', 'pos'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.figure()
    plt.show()