# References:
    # https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np
import cv2

from celeba import CelebADS
from utils import image_to_grid, to_pil
from quantize3 import test_opencv


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def simulate_user_stroke_input(image, kernel_size=3, n_colors=10):
    med_filter = MedianPool2d(kernel_size=kernel_size, stride=1, padding=1)

    image = med_filter(image[None, ...])
    grid = image_to_grid(image, n_cols=1)
    quantized_grid = test_opencv(
        np.array(grid), cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB, n_colors=n_colors,
    )
    return to_pil(quantized_grid)


if __name__ == "__simulate_user_stroke_input__":
    data_dir = "/Users/jongbeomkim/Documents/datasets"
    img_size = 64
    ds = CelebADS(
        data_dir=data_dir, split="test", img_size=img_size, hflip=False,
    )
    image = ds[3]
    image = simulate_user_stroke_input(image)
    image.show()

    # from PIL.ImageFilter import MedianFilter
    
    # med_filter = MedianFilter(size=23)
    # image = med_filter.filter(image[None, ...])
