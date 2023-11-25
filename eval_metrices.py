import matplotlib.pyplot as plt
import os
import sklearn.metrics


def roc_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=''):
    assert positive_label == 1

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions,
                                            pos_label=positive_label)

    auc = sklearn.metrics.roc_auc_score(labels, predictions) # TODO consider replace with metrics.auc(fpr, tpr) since it has the label built in implicit
    print("AUC: {}".format(auc))
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f) support = %3d' % (auc, n_labels))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = int(thresholdsLength/thresholds_every)
    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(fpr[i] - 0.03, tpr[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'roc_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")


def p_r_plot(labels, predictions, positive_label, save_dir, thresholds_every=5, unique_id=None):
    assert positive_label == 1

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                pos_label=positive_label)

    ap = sklearn.metrics.average_precision_score(labels, predictions,
                                                                pos_label=positive_label)

    print("AP : {}".format(ap))
    # auc = sklearn.metrics.roc_auc_score(labels, predictions)
    granularity_percentage = 1. / labels.shape[0] *100
    lw = 2
    n_labels = len(labels)

    plt.figure()
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='AP (area = %0.3f) support = %3d' % (ap, n_labels))
    plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR model {} (gran={:.2e}[%])".format(unique_id, granularity_percentage))
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholdsLength = len(thresholds) #- 1
    thresholds_every = max(int(thresholdsLength/thresholds_every), 1)

    thresholds = thresholds[1:] #  `thresholds[0]` represents no instances being predicted and is arbitrarily set to `max(y_score) + 1`. https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    thresholdsLength = thresholdsLength - 1 #changed the threshold vector henc echange the len

    colorMap = plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = thresholds[i].__format__('.3f')
        plt.text(recall[i] - 0.03, precision[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15},
                 color=colorMap(i / thresholdsLength))

    filename = unique_id + 'p_r_curve.png'
    plt.savefig(os.path.join(save_dir, filename), format="png")

