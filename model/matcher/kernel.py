# This file is part of EquivAlign.
# 
# Copyright [2024] [Authors of Paper: Correspondence-free SE(3) point cloud registration in RKHS via unsupervised equivariant learning]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author Email: <Ray Zhang rzh@umich.edu>

#import numpy
import torch
import torch.nn as nn
#import torch.utils.data
import torch.nn.functional as F
#from model import option
from collections import OrderedDict
import pdb


class GaussianKernel(nn.Module):
    def __init__(self, init_ell, requires_grad=True):
        super(GaussianKernel, self).__init__()
        self.init_ell = init_ell
        self.requires_grad = requires_grad
        if (requires_grad == False):
            self.register_buffer('ell', torch.ones(1).cuda() * init_ell )
            
        else:
            self.register_parameter('ell', torch.nn.Parameter(torch.ones(1).cuda() * init_ell) )            

    def initialize_kernel(self):
        if self.requires_grad:
            if self.ell is None:
                self.register_parameter('ell', torch.nn.Parameter(torch.ones(1).cuda() * self.init_ell) )
            else:
                self.ell = torch.nn.Parameter((torch.ones(1).cuda() * self.init_ell).requires_grad_(True))


    ### input shape: [batch, num_points, flattned_feature_vec]
    def forward(self, x_flat, y_flat):
        dist = torch.cdist(x_flat, y_flat, 2)

        return torch.exp(-torch.mul(dist, dist) / 2 / (torch.clamp(self.ell, min=0.01) * torch.clamp(self.ell, min=0.01)))

class CosKernel(nn.Module):
    def __init__(self) -> None:
        super(CosKernel, self).__init__()

    def initialize_kernel(self):
        pass #self.register_parameter('ell', torch.nn.Parameter(torch.ones(1).cuda() * init_ell) )            
        

    ### input shape: [batch, num_points, flattned_feature_vec]
    def forward(self, x_flat, y_flat):
        #dist = torch.cdist(x_flat, y_flat, 2)
        #pdb.set_trace()
        cos_val = x_flat @ y_flat.permute(0, 2, 1)
        #cos_val = torch.einsum('bnc,bmc->bnm', x_flat, y_flat)
        norm_x = torch.norm(x_flat, p=2, dim=2).unsqueeze(2).expand(x_flat.shape[0], x_flat.shape[1], y_flat.shape[1])#torch.einsum('bnc,bmc->bnm', x_flat, x_flat)
        norm_y = torch.norm(y_flat, p=2, dim=2).unsqueeze(2).expand(y_flat.shape[0], x_flat.shape[1], y_flat.shape[1])#torch.einsum('bnc,bmc->bnm', y_flat, y_flat)
        #pdb.set_trace()
        cos_val = cos_val / norm_x / (norm_y)
        #ret = cos_val
        ret = torch.tanh(cos_val) + 1.0  #torch.exp(-torch.mul(dist, dist) / 2 / (self.ell * self.ell))
        #pdb.set_trace()
        return ret

class TanhKernel(nn.Module):
    def __init__(self) -> None:
        super(TanhKernel, self).__init__()

    def initialize_kernel(self):
        pass
        
    ### input shape: [batch, num_points, flattned_feature_vec]
    def forward(self, x_flat, y_flat):
        pdb.set_trace()
        x_flat_expand = x_flat.unsqueeze(2).expand(x_flat.shape[0],
                                                   x_flat.shape[1],
                                                   x_flat.shape[2],
                                                   x_flat.shape[2])
        
    
KERNEL_MAP ={
    'GaussianKernel': GaussianKernel,
    'CosKernel': CosKernel,
    'TanhKernel': TanhKernel
}

def gen_kernel(kernel_type: str,
               init_ell: float,
               is_learning: bool):
    assert (kernel_type in KERNEL_MAP)
    #return KERNEL_MAP[kernel_type](init_ell, is_learning)
    if (kernel_type == 'GaussianKernel'):
        return GaussianKernel(init_ell, is_learning)
    elif kernel_type == 'CosKernel':
        return CosKernel()
    elif kernel_type == 'TanhKernel':
        return TanhKernel()
