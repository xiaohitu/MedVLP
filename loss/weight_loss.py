import torch
import torch.nn as nn


def log_sum_exp(x):
    # See implementation detail in
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # b is a shift factor. see link.
    # x.size() = [N, C]:
    # import pdb; pdb.set_trace()
    b, _ = torch.max(x, 1)
    b = torch.unsqueeze(torch.squeeze(b), dim=1)
    y = b + torch.log(torch.exp(x - b.repeat(1, x.size(1))).sum(1, True))
    # y.size() = [N, 1]. Squeeze to [N] and return
    return y.squeeze(1)


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    # import pdb; pdb.set_trace()
    loss = log_sum_exp(logits) - class_select(logits, target)

    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """

    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()

    # alpha:3*150
    # logits:450*2
    # target:450
    # batch_size:3
    def forward(self, alpha, logits, target):
        softmax_logits = torch.softmax(logits, dim=1)
        # print('softmax_logits', softmax_logits.shape)
        # selected_logits:450*1->3*150
        temp = class_select(softmax_logits, target)
        selected_logits = temp.view(alpha.shape[0], -1)
        # print('selected_logits', selected_logits.shape)
        logit_loss = torch.norm(input=alpha - selected_logits, p=2, dim=-1).sum()
        # print('logit_loss', logit_loss)

        return logit_loss
