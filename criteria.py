import torch


"""
DSC based losses
"""


def gendsc_loss(pred, target, w_bg=None, w_fg=None):
    """
    Function to compute the generalised Dice loss based on:
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge
    Cardoso, "Generalised Dice Overlap as a Deep Learning Loss Function for
    Highly Unbalanced Segmentations".
    https://arxiv.org/abs/1707.03237
    https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, data_shape]
    :param target: Ground truth values. This tensor should have the shape:
     [batch_size, data_shape]
    :param w_bg: Weight given to background voxels.
    :param w_fg: Weight given to foreground voxels.
    :return: The DSC loss for the batch
    """
    # Init
    # Dimension checks. We want everything to be the same. This a class vs
    # class comparison.
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    loss = []
    for y, y_hat in zip(target, pred):
        m_bg = y == 0
        m_fg = y > 0
        n_bg = torch.sum(m_bg)
        n_fg = torch.sum(m_fg)
        n = torch.numel(y)

        if w_bg is None:
            if n_bg > 0:
                w_bg = torch.sqrt(n_bg.type(torch.float32)) ** -2
            else:
                w_bg = 0
        if w_fg is None:
            if n_fg > 0:
                w_fg = torch.sqrt(n_fg.type(torch.float32)) ** -2
            else:
                w_fg = 0

        sum_pred_fg = torch.sum(y_hat[m_fg])
        sum_pred = torch.sum(y_hat)

        tp_term = (w_fg + w_bg) * sum_pred_fg
        tn_term = w_bg * (n_bg - sum_pred)
        den = (w_fg - w_bg) * (n_fg + sum_pred) + 2 * n * w_bg

        if w_bg == 0 and den == 0:
            den = den + 1e-6

        loss.append(1 - 2 * (tp_term + tn_term) / den)
    loss = sum(loss) / len(loss)

    return loss


def new_loss(pred, target, weight_bg=None, weight_fg=None, gamma=2):
    return WeightedLoss.apply(pred, target, weight_bg, weight_fg, gamma)


class WeightedLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pred, target, weight_bg=None, weight_fg=None, gamma=2):
        m_bg = target == 0
        m_fg = target > 0
        y_hat = torch.sigmoid(pred)

        if weight_bg is None:
            weight_bg = float(torch.sum(m_fg)) / torch.numel(target)
        if weight_fg is None:
            weight_fg = float(torch.sum(m_bg)) / torch.numel(target)

        ctx.save_for_backward(y_hat, target)
        ctx.weight_bg = weight_bg
        ctx.weight_fg = weight_fg
        ctx.gamma = gamma

        loss = gendsc_loss(y_hat, target)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        pred_y, target = ctx.saved_tensors

        negative = (1 - target) * pred_y ** ctx.gamma
        positive = target * (1 - pred_y) ** ctx.gamma
        dsigmoid = ctx.weight_bg * negative - ctx.weight_fg * positive

        grad_input = grad_output * dsigmoid

        return grad_input, None, None, None, None


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
    if len(dsc_k) > 0:
        dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = torch.tensor(0)

    return torch.clamp(dsc, 0., 1.)


def tp_binary_loss(pred, target):
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
        torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    tp_k = intersection / sum_target
    tp_k = tp_k[torch.logical_not(torch.isnan(tp_k))]
    tp = 1 - torch.mean(tp_k)

    return torch.clamp(tp, 0., 1.)


def tn_binary_loss(pred, target):
    pred = torch.flatten(pred < 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            torch.sum(pred & torch.logical_not(target), dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        torch.logical_not(target), dim=1
    ).type(torch.float32).to(pred.device)

    tn_k = intersection / sum_target
    tn_k = tn_k[torch.logical_not(torch.isnan(tn_k))]
    tn = 1 - torch.mean(tn_k)

    return torch.clamp(tn, 0., 1.)
