import torch


"""
DSC based losses
"""


def dsc_loss(pred, target, eps=1e-4):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param eps: Parameter used to prevent numerical errors when no positive
     samples are available.
    :return: The mean DSC for the batch
    """
    # Init
    dims = pred.shape
    # Dimension checks. We want everything to be the same. This a class vs
    # class comparison.
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    # We'll do the sums / means across the 3D axes to have a value per patch.
    # There is only a class here.
    # DSC = 2 * | pred *intersection* target | / (| pred | + | target |)
    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims).clamp(min=eps)
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


def dsc_binary_loss(pred, target):
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            2 * torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_pred = torch.sum(pred, dim=1).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    dsc_k = intersection / (sum_pred + sum_target)
    dsc_k = dsc_k[torch.logical_not(torch.isnan(dsc_k))]
    print(dsc_k)
    if len(dsc_k) > 0:
        dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = 0

    return torch.clamp(dsc, 0., 1.)
