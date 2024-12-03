from __future__ import print_function, division, absolute_import
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from ..ops import CTA
from ..ops import TTR
from ..ops.MS_ST3D import MS_ST3D

import torch.nn as nn
from mmcv.cnn import  kaiming_init
from mmcv.runner import load_checkpoint

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES

@BACKBONES.register_module()
class Net(MS_ST3D):
    def __init__(self,
                 in_channels=512,
                 groups=(10,6),
                 squeeze_factor=1,
                 pretrained=None):
        super(Net, self).__init__(in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(3, 4, 6),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2))
        self.pretrained=pretrained
        self.CTA = CTA(in_channels, groups, squeeze_factor)

        self.TTR=TTR(in_channels,in_channels)

    def forward(self, input):  #4 17 48 56 56
        x = super(Net, self).forward(input)  # 3dcnn:4 512 24 6 6

        output1 = self.CTA(x)#4 512 24 6 6

        output2 = self.TTR(output1)  # 4 512 24 6 6

        output = output2+x # 4 512 24  6 6
        return    output
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
