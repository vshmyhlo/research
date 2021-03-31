import numpy as np
from matplotlib import pyplot as plt


def compute_errors(input, target):
    sort = np.argsort(-input)
    target = target[sort]
    thresholds = input[sort]

    tp = np.cumsum(target)
    tn = flip_cumsum_shift_flip(1 - target)
    fp = np.cumsum(1 - target)
    fn = flip_cumsum_shift_flip(target)

    result = tp, tn, fp, fn, thresholds

    i = np.where(np.diff(thresholds))[0]
    i = np.r_[i, len(thresholds) - 1]
    result = [x[i] for x in result]

    return result


def precision_recall_curve(input, target):
    tp, tn, fp, fn, thresholds = compute_errors(input, target)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision = np.concatenate([[1], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    # TODO: fix t_min and t_max
    thresholds = np.concatenate([[thresholds.max() + 1], thresholds, [thresholds.min() - 1]])

    return precision, recall, thresholds


def precision_recall_auc(input, target):
    precision, recall, _ = precision_recall_curve(input, target)
    auc = np.trapz(precision, recall)

    return auc


def plot_pr_curve(input, target):
    precision, recall, _ = precision_recall_curve(input, target)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.fill_between(recall, 0, precision, alpha=0.1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig


def flip_cumsum_shift_flip(x):
    x = np.flip(x)
    x = np.cumsum(x)
    x = np.r_[0, x[:-1]]
    x = np.flip(x)

    return x
