from collections import OrderedDict

import torch
import torch.nn.functional as F
from timm import models
from torch import nn


class FocalNet(nn.Module):
    def __init__(self, level, fork_feat, pretrained=False):
        super(FocalNet, self).__init__()
        self.fork_feat = fork_feat
        if level == 'focalnet_base_srf':
            backbone = models.focalnet_base_srf(pretrained=pretrained)
        elif level == 'focalnet_base_lrf':
            backbone = models.focalnet_base_lrf(pretrained=pretrained)
        elif level == 'focalnet_huge_fl4':
            backbone = models.focalnet_huge_fl4(pretrained=pretrained)
        elif level == 'focalnet_huge_fl3':
            backbone = models.focalnet_huge_fl3(pretrained=pretrained)
        elif level == 'focalnet_large_fl4':
            backbone = models.focalnet_large_fl4(pretrained=pretrained)
        elif level == 'focalnet_large_fl3':
            backbone = models.focalnet_large_fl3(pretrained=pretrained)
        elif level == 'focalnet_small_srf':
            backbone = models.focalnet_small_srf(pretrained=pretrained)
        elif level == 'focalnet_small_lrf':
            backbone = models.focalnet_small_lrf(pretrained=pretrained)
        elif level == 'focalnet_tiny_srf':
            backbone = models.focalnet_tiny_srf(pretrained=pretrained)
        elif level == 'focalnet_tiny_lrf':
            backbone = models.focalnet_tiny_lrf(pretrained=pretrained)
        elif level == 'focalnet_xlarge_fl4':
            backbone = models.focalnet_xlarge_fl4(pretrained=pretrained)
        elif level == 'focalnet_xlarge_fl3':
            backbone = models.focalnet_xlarge_fl3(pretrained=pretrained)

        self.stem = backbone.stem
        self.layers = backbone.layers

    def forward(self, x):
        x = self.stem(x)
        out = {}
        for i, block in enumerate(self.layers):
            x = block(x)
            if (i + 2) in self.fork_feat:
                out[f'p{(i+2)}'] = x
        return OrderedDict(out)


def main():
    levels = ['focalnet_base_srf', 'focalnet_base_lrf', 'focalnet_huge_fl4', 'focalnet_huge_fl3', 'focalnet_large_fl4', 'focalnet_large_fl3', 'focalnet_small_srf',
              'focalnet_small_lrf', 'focalnet_tiny_srf', 'focalnet_tiny_lrf', 'focalnet_xlarge_fl4', 'focalnet_xlarge_fl3']
    for level in levels:
        backbone = FocalNet(level, [2, 3, 4, 5])
        a = torch.randn(1, 3, 800, 800)
        res = backbone(a)
        print(level)
        for key in res:
            print(res[key].shape[1])
        print()


if __name__ == "__main__":
    main()
