# References:
    # https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    # https://github.com/Chadys/QuantizeImageMethods/blob/master/demo_quantize_methods.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

from celeba import CelebADS
from utils import image_to_grid


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()

        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding) # Convert to LTRB.
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
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class StrokeSimulator(object):
    def __init__(self, kernel_size):
        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2(),
            ]
        )
        self.med_filter = MedianPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
        )

    @staticmethod
    def update_n_colors(label, n_pixels, thresh_pixel_per, n_colors):
        n_colors_under_thresh = 0
        label = label.flatten()
        color_count = Counter(label)
        for (_, count) in color_count.items():
            if count / n_pixels < thresh_pixel_per:
                n_colors_under_thresh += 1
        n_colors -= -(-n_colors_under_thresh // 2) # Ceil integer division.
        return n_colors, n_colors_under_thresh

    @staticmethod
    def process_result(center, label, shape, conv_method):
        center = center.astype("uint8")
        quantized_img = center[label]
        quantized_img = quantized_img.reshape(shape)
        quantized_img = cv2.cvtColor(quantized_img, conv_method)
        center = cv2.cvtColor(np.expand_dims(center, axis=0), conv_method)[0]
        return quantized_img


    def quantize_img(
        self, img, n_colors, method1=cv2.COLOR_RGB2Lab, method2=cv2.COLOR_Lab2RGB,
    ):
        img = cv2.cvtColor(np.array(img), method1)
        flat_img = img.reshape((-1, 3)).astype("float32")
        n_pixels = img.size

        thresh_pixel_per = 0.01
        n_colors_under_thresh = n_colors
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0,
        )
        center = None
        label = None

        while n_colors_under_thresh > 0:
            _, label, center = cv2.kmeans(
                flat_img, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            n_colors, n_colors_under_thresh = self.update_n_colors(
                label, n_pixels, thresh_pixel_per, n_colors  # , flat_img
            )
        return self.process_result(center, label, img.shape, method2)

    def quantize_image_tensor(self, image, n_colors):
        new_batches = list()
        for batch in torch.chunk(image, chunks=image.size(0), dim=0):
            image = image_to_grid(batch, n_cols=1)
            img = np.array(image)
            quantized_img = self.quantize_img(img, n_colors=n_colors)
            quantized_image = self.transform(image=quantized_img)["image"]
            new_batches.append(quantized_image)
        return torch.stack(new_batches, dim=0)

    def __call__(self, image, n_colors=10):
        image = self.med_filter(image)
        return self.quantize_image_tensor(image, n_colors=n_colors)


if __name__ == "__main__":
    data_dir = "/home/dmeta0304/Documents/datasets/"
    img_size = 64

    ds = CelebADS(
        data_dir=data_dir, split="test", img_size=img_size, hflip=False,
    )
    image = torch.stack([ds[0], ds[1], ds[2], ds[3]], dim=0)

    n_colors = 3
    stroke_sim = StrokeSimulator(kernel_size=3)
    quant_image = stroke_sim(image, n_colors=n_colors)
    quant_grid = image_to_grid(quant_image, n_cols=1)
    quant_grid.show()
