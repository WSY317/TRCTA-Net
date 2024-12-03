# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """
    # 用一系列转换组成一个数据管道。
    #
    # 参数:
    # 转换(list[dict | callable]):
    # 要么配置转换的字典，要么配置转换对象。
    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:#在Compose初始化中，通过遍历transforms里面的8个元素，利用build_from_cfg函数完成了各个类的实例化，
                                    # 之后将各个实例对象append进self.transforms列表中。至此，Compose类实际上里面存储的是顺序的图像增强实例对象。
                                    # 至此，poseDataset初始化部分完成。
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)#t=uniformsampleFeames,data是上一个管道的results
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
