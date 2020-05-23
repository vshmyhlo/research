import torch


def pad_and_pack(tensors):
    sizes = [t.shape[0] for t in tensors]

    tensor = torch.zeros(
        len(sizes), max(sizes), dtype=tensors[0].dtype, layout=tensors[0].layout, device=tensors[0].device)
    mask = torch.zeros(
        len(sizes), max(sizes), dtype=torch.bool, layout=tensors[0].layout, device=tensors[0].device)

    for i, t in enumerate(tensors):
        tensor[i, :t.size(0)] = t
        mask[i, :t.size(0)] = True

    return tensor, mask


def collate_fn(batch):
    sigs, syms = list(zip(*batch))

    sigs, sigs_mask = pad_and_pack(sigs)
    syms, syms_mask = pad_and_pack(syms)

    return (sigs, syms), (sigs_mask, syms_mask)
