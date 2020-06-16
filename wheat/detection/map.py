from functools import reduce

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from detection.box_utils import boxes_pairwise_iou


# TODO: refactor


def build_sample_matches(boxes_true, boxes_pred, scores_pred, iou_threshold):
    argsort = scores_pred.argsort(descending=True)
    boxes_pred = boxes_pred[argsort]
    scores_pred = scores_pred[argsort]

    ious = boxes_pairwise_iou(boxes_pred, boxes_true)
    match = torch.zeros(ious.size(0), dtype=torch.bool)

    if ious.size(1) > 0:
        for i in range(ious.size(0)):
            v, j = ious[i].max(0)
            if v > iou_threshold:
                match[i] = True
                ious[:, j] = 0.

    sample_matches = pd.DataFrame({
        'score': scores_pred.data.cpu().numpy(),
        'match': match.data.cpu().numpy(),
    })

    return sample_matches, boxes_true.size(0)


def build_per_class_sample_matches(true, pred, iou_threshold):
    class_ids = torch.cat([true.class_ids, pred.class_ids], 0).unique()

    per_class_matches = []
    per_class_num_true = pd.Series([0] * class_ids.size(0), index=class_ids.data.cpu().numpy())

    for class_id in class_ids:
        sample_matches, per_class_num_true.loc[class_id.item()] = build_sample_matches(
            true.boxes[true.class_ids == class_id],
            pred.boxes[pred.class_ids == class_id],
            pred.scores[pred.class_ids == class_id],
            iou_threshold=iou_threshold)
        sample_matches['class_id'] = class_id.item()
        per_class_matches.append(sample_matches)

    if len(per_class_matches) > 0:
        per_class_matches = pd.concat(per_class_matches)
    else:
        per_class_matches = pd.DataFrame({
            'score': np.zeros([0], dtype=np.float),
            'match': np.zeros([0], dtype=np.bool),
            'class_id': np.zeros([0], dtype=np.int64)
        })
        assert per_class_matches.empty

    return per_class_matches, per_class_num_true


def per_class_precision_recall(batch_true, batch_pred, iou_threshold):
    assert len(batch_true) == len(batch_pred)

    sample_results = [
        build_per_class_sample_matches(true, pred, iou_threshold=iou_threshold)
        for true, pred in zip(batch_true, batch_pred)]

    per_class_matches, per_class_num_true = zip(*sample_results)
    per_class_matches = pd.concat(per_class_matches).sort_values('score', ascending=False)
    per_class_num_true = reduce(lambda a, b: a.add(b, fill_value=0.), per_class_num_true)

    pr = {}
    for class_id, group in per_class_matches.groupby('class_id'):
        match = torch.tensor(group['match'].values)
        pr[class_id] = match_to_precision_recall(match, per_class_num_true.loc[class_id])

    return pr


def per_class_precision_recall_to_map(pr):
    map = [pr_auc(pr[k]) for k in pr]
    map = torch.tensor(map).mean()

    return map


def pr_auc(pr):
    pr = F.pad(pr, (0, 0, 2, 2))
    pr[[1, -2], 1] = pr[[2, -3], 1]
    pr[-1, 1] = 1.

    pr = pr.data.cpu().numpy()
    auc = np.trapz(pr[:, 0], pr[:, 1])

    return auc


def match_to_precision_recall(match, num_true):
    cumsum = match.float().cumsum(0)
    precision = cumsum / torch.arange(1, match.size(0) + 1, dtype=torch.float)
    recall = cumsum / num_true

    return torch.stack([precision, recall], 1)


if __name__ == '__main__':
    true, pred = torch.load('./tmp.pth', map_location='cpu')
    true = [t[:2] for t in true]
    pr = per_class_precision_recall(true, pred, 0.5)
    print(per_class_precision_recall_to_map(pr))
