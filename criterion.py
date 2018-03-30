from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    https://arxiv.org/pdf/1708.02002.pdf

    NOTE: this alpha overweights positive classes.
    This creates a loss magnitude more equal with the bbox loss
    """

    def __init__(self, gamma=2, alpha=1e3, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        batch_size, num_anchors, num_classes = outputs.size()
        alpha = outputs.data.new(num_classes).fill_(1)
        alpha[1:].fill_(self.alpha)
        outputs = outputs.view(-1, num_classes)
        P = F.softmax(outputs, dim=1).max(dim=1)[0]
        cross_entropy = F.cross_entropy(outputs, targets.view(-1),
                                        ignore_index=self.ignore_index,
                                        reduce=True)
        return cross_entropy
        return ((1.0 - P) ** self.gamma * cross_entropy).mean()
