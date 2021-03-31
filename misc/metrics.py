import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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


def far_tar_curve(input, target):
    tp, tn, fp, fn, thresholds = compute_errors(input, target)

    far = fn / (fn + tp)
    tar = tn / (tn + fp)

    far = np.concatenate([[1], far, [0]])
    tar = np.concatenate([[1], tar, [0]])
    # TODO: fix t_min and t_max
    thresholds = np.concatenate([[thresholds.max() + 1], thresholds, [thresholds.min() - 1]])

    return far, tar, thresholds


def tar_at_far(input, target, at_far):
    far, tar, thresholds = far_tar_curve(input, target)

    values = np.interp(at_far, np.flip(far), np.flip(tar))
    thresholds = np.interp(at_far, np.flip(far), np.flip(thresholds))

    return values, thresholds


def far_tar(input, target):
    tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=input).ravel()

    far = fn / (fn + tp)
    tar = tn / (tn + fp)

    return far, tar


def far_tar_auc(input, target):
    far, tar, _ = far_tar_curve(input, target)
    indices = np.s_[::-1]
    auc = np.trapz(tar[indices], far[indices])

    return auc


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


def plot_far_tar_curve(input, target):
    far, tar, _ = far_tar_curve(input, target)

    fig = plt.figure()
    plt.plot(far, tar)
    plt.fill_between(far, 0, tar, alpha=0.1)
    plt.xlabel("far")
    plt.ylabel("tar")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig


def plot_confusion_matrix(
    input,
    target,
    *,
    labels=None,
    sample_weight=None,
    normalize=None,
    include_values=True,
    xticks_rotation="horizontal",
    values_format=None,
    cmap="viridis",
    ax=None
):
    cm = confusion_matrix(
        y_true=target,
        y_pred=input,
        sample_weight=sample_weight,
        normalize=normalize,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp = disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        values_format=values_format,
    )

    return disp.figure_


def flip_cumsum_shift_flip(x):
    x = np.flip(x)
    x = np.cumsum(x)
    x = np.r_[0, x[:-1]]
    x = np.flip(x)

    return x
