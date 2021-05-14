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
    # DSC = 2 * | pred *union* target | / (| pred | + | target |)
    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims).clamp(min=eps)
    dsc_k = num / den
    if torch.isnan(dsc_k).any():
        print(dsc_k, num, den)
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)
