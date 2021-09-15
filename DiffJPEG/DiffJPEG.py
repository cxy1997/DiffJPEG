# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local
from .modules import compress_jpeg, decompress_jpeg
from .utils import diff_round, quality_to_factor, DiffRound


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, subsample=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image height
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        self.height, self.width = height, width
        if differentiable:
            # rounding = DiffRound()
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor, subsample=subsample)
        self.decompress = decompress_jpeg(height, width, rounding=rounding, subsample=subsample, factor=factor)

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

    def redistribute(self, core):
        height, width = self.height // 8, self.width // 8
        size = core.shape[2] // height
        core = core.contiguous().view(core.shape[0], 3, height, size, width, size).permute(0, 1, 2, 4, 3, 5).contiguous().view(core.shape[0], 3, height * width, size, size)
        y, cb, cr = torch.split(F.pad(core, (0, 8 - size, 0, 8 - size), "constant", 0), 1, dim=1)
        return y.squeeze(1), cb.squeeze(1), cr.squeeze(1)

    def merge_block(self, patches):
        height, width = self.height // 8, self.width // 8
        batch_size = patches.shape[0]
        k = patches.shape[2]
        image_reshaped = patches.view(batch_size, height, width, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, 1, height * k, width * k)

    def extract_freq(self, x, size):
        y, cb, cr = self.compress(x, quantize=False)
        return torch.cat([self.merge_block(y[:, :, :size, :size]),
                          self.merge_block(cb[:, :, :size, :size]),
                          self.merge_block(cr[:, :, :size, :size])],
                         dim=1)

    def quantize(self, x):
        height, width = self.height // 8, self.width // 8
        size = x.shape[2] // height
        y, cb, cr = self.redistribute(x)
        y, cb, cr = self.compress.quantize(y, cb, cr)
        return torch.cat([self.merge_block(y[:, :, :size, :size]),
                          self.merge_block(cb[:, :, :size, :size]),
                          self.merge_block(cr[:, :, :size, :size])],
                         dim=1)

    def recover(self, x):
        y, cb, cr = self.redistribute(x)
        return self.decompress(y, cb, cr)
