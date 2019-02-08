from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(res_dir, y_val, y_val_results):
    false_pos_rate, true_pos_rate, threshold = roc_curve(y_val, y_val_results)
    auc = roc_auc_score(y_val, y_val_results)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_pos_rate, true_pos_rate, label='AUC=%0.2f' % auc)
    plt.legend(loc='lower right')
    plt.savefig(res_dir + '/roc.pdf')