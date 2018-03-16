from torch import nn
from torchvision.models import resnet50


class PyramidLayer(nn.Module):
    """
    Combines coarse features
    with higher granularity features from backbone network.
    """

    def __init__(self, in_channels, out_channels):
        super(PyramidLayer, self).__init__()  # init parent properties
        self.lateral_pathway = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.anti_alias_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, coarse_features, encoder_channel):
        upsampled = self.upsample(coarse_features)
        squeezed = self.lateral_pathway(encoder_channel)
        return self.anti_alias_conv(upsampled + squeezed)


class FPN50(nn.Module):
    """
    Converts Images into four multiscale feature maps.

    RetinaNet version of FPN

    Focal Loss for Dense Object Detection
    https://arxiv.org/pdf/1708.02002.pdf

    Feature Pyramid Networks for Object Detection
    https://arxiv.org/pdf/1612.03144.pdf
    """

    def __init__(self, pretrained=False):
        super(FPN50, self).__init__()  # init parent properties
        resnet = resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_layer1 = resnet.layer1
        self.down_layer2 = resnet.layer2
        self.down_layer3 = resnet.layer3
        self.down_layer4 = resnet.layer4

        self.p7_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.p6_conv = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.pyramid_4 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.pyramid_3 = PyramidLayer(in_channels=1024, out_channels=256)
        self.pyramid_2 = PyramidLayer(in_channels=512, out_channels=256)
        self.pyramid_1 = PyramidLayer(in_channels=256, out_channels=256)

    def forward(self, x):
        # with input 256, 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.down_layer1(x)  # (64, 64) stride = 4
        c3 = self.down_layer2(c2)  # (32, 32) stride = 8
        c4 = self.down_layer3(c3)  # (16, 16) stride = 16
        c5 = self.down_layer4(c4)  # (8, 8) stride = 32

        # retinanet modifications
        p6 = self.p6_conv(c5)  # stride 64?
        p7 = self.relu(self.p7_conv(p6))  # stride 128

        p5 = self.pyramid_4(c5)  # (8, 8) coarsest feature map
        p4 = self.pyramid_3(p5, c4)
        p3 = self.pyramid_2(p4, c3)
        #         p2 = self.pyramid_1(p3, c2) # retinanet removes p2
        return p3, p4, p5, p6, p7
