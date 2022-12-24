
import torch
from torch import nn
import torchvision.models as models


# class Resnet18(nn.Module):

#     def __init__(self, n_classes):
#         super(Resnet18, self).__init__()
#         self.net = models.resnet18(pretrained=True)

#         layer0 = nn.Sequential(
#                         self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool
#                         )
#         layer1 = nn.Sequential(
#                         layer0,
#                        self.net.layer1,
#         )
#         layer2 = self.net.layer2
#         layer3 = self.net.layer3
#         layer4 = self.net.layer4

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.softmax = nn.Softmax(dim=1)
#         self.fc = nn.Linear(layer4[-1].conv2.out_channels, n_classes)

#         self.net = nn.Sequential(
#                         layer1,  #1
#                         layer2,  #2
#                         layer3,  #3
#                         nn.Sequential(layer4, self.avgpool, nn.Flatten(), self.fc)  #4
#                         )


#     def forward(self, input, layer_num=None):
#         if layer_num != None:
#             feature = self.net[layer_num](input)
#         else:
#             feature = self.net(input)
#         return feature


class Resnet18_k8(nn.Module):

    def __init__(self, n_classes):
        super(Resnet18_k8, self).__init__()
        self.net = models.resnet18(pretrained=True)

        layer0 = nn.Sequential(
                        self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool
                              )
                              
        # Resblocks (#blocks x memory consumption)
        # layer1: 3 X 8 into 3 groups of 1x8
        layer1_1 = self.net.layer1[0]
        layer1_2 = self.net.layer1[1]

        # layer2: 4 X 4 into 2 groups of 2x4
        layer2_1 = self.net.layer2[0]
        layer2_2 = self.net.layer2[1]

        # layer3: 6 X 2 into 2 groups of 3x2
        layer3_1 = self.net.layer3[0]
        layer3_2 = self.net.layer3[1]

        # layer4: 3 X 1 unchanged
        layer4_1 = self.net.layer4[0]  # 3 X 1
        layer4_2 = self.net.layer4[1] 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(layer4_2.conv2.out_channels, n_classes)

        self.net = nn.Sequential(
                        layer0,    #1
                        layer1_1,  #2
                        layer1_2,  #3
                        layer2_1,  #5
                        layer2_2,  #6
                        layer3_1,  #7
                        layer3_2,  #8
                        layer4_1,  #9
                        nn.Sequential(layer4_2, self.avgpool, nn.Flatten()),  #10
                        self.fc
                        )


    def forward(self, input, layer_num=None):
        if layer_num != None:
            feature = self.net[layer_num](input)
        else:
            feature = self.net(input)
        return feature
    