import torch
from torch import nn

from .fpn import FPN50


class Subnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Subnet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.relu(self.conv4(x))


class RetinaNet(nn.Module):
    """
    Single-stage Object detection network, outputs dense class and bounding box predictions
    size [batch, num_features*num_anchors, num_classes] and [batch, num_features*num_anchors, 4]
    num_features = sum of all feature map areas from fpn.
    """

    def __init__(self, num_classes, num_anchors=9):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.classification_subnet = Subnet(in_channels=256, out_channels=self.num_anchors * num_classes)
        self.box_regression_subnet = Subnet(in_channels=256, out_channels=self.num_anchors * 4)

    def forward(self, x):
        feature_pyramid = self.fpn(x)
        class_predictions = []
        box_predictions = []
        for features in feature_pyramid:
            class_prediction = self.classification_subnet(features)
            box_prediction = self.box_regression_subnet(features)
            # permute preds to (batch, h, w, anchor*prediction),
            # then flatten to (batch, h*w*num_anchor, num_class) for stacking/loss
            batch_size = class_prediction.size(0)
            class_prediction = class_prediction.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            box_prediction = box_prediction.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            class_predictions.append(class_prediction)
            box_predictions.append(box_prediction)
        return torch.cat(class_predictions, dim=1), torch.cat(box_predictions, dim=1)
