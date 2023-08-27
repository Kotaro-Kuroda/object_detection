from torch import nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
import torch


class EfficientNet(nn.Module):
    def __init__(self, level, fork_feat, pretrained=False):
        super(EfficientNet, self).__init__()
        self.fork_feat = fork_feat
        if level == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        elif level == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b1(pretrained)
        elif level == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b2(pretrained)
        elif level == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(pretrained)
        elif level == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b4(pretrained)
        elif level == 'efficientnet_b5':
            weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b5(pretrained)
        elif level == 'efficientnet_b6':
            weights = models.EfficientNet_B6_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b6(pretrained)
        elif level == 'efficientnet_b7':
            weights = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b7(pretrained)
        elif level == 'efficientnet_b8':
            weights = models.EfficientNet_B8_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b8(pretrained)
        elif level == 'efficientnetv2_m':
            weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_m(pretrained)
        elif level == 'efficientnetv2_l':
            weights = models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_l(pretrained)
        else:
            raise ValueError()
        self.layer0 = backbone.features[0]
        self.blocks = backbone.features[1:-1]
        self.layer1 = self.blocks[0]
        self.layer2 = self.blocks[1]
        self.layer3 = self.blocks[2]
        self.layer4 = self.blocks[3]
        self.layer5 = self.blocks[4]
        self.layer6 = self.blocks[5]
        self.layer7 = self.blocks[6]

    def forward(self, x):
        x = self.layer0(x)
        out = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) in self.fork_feat:
                out[f'p{(i+1)}'] = x
        return OrderedDict(out)


def main():
    for i in range(8):
        level = f'efficientnet_b{i}'
        backbone = EfficientNet(level, [0, 1, 2, 3, 4, 5, 6, 7])
        a = torch.randn(1, 3, 1024, 1024)
        out = backbone(a)
        print(level)
        for key in out.keys():
            print(out[key].shape)
        print()


if __name__ == "__main__":
    main()
