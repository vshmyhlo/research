import torch
import torchvision


def per_class_nms(boxes, scores, class_ids, iou_threshold):
    mask = torch.zeros(boxes.size(0), dtype=torch.bool)

    for id in class_ids.unique():
        subset_mask = class_ids == id
        keep_mask = torch.zeros(subset_mask.sum(), dtype=torch.bool)

        keep = torchvision.ops.nms(boxes[subset_mask], scores[subset_mask], iou_threshold)

        keep_mask[keep] = True
        mask[subset_mask] = keep_mask

    (keep,) = torch.where(mask)

    return keep
