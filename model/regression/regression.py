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
#import torch
#import torch.nn as nn
#import pypose as pp
#import pypose.optim as ppos
#from option import opt
from .inner_product_optimization import InnerProductOptimization

REGRESSION_MAP = {
    'InnerProductOptimization': InnerProductOptimization
}
def gen_regression(matcher, opt):
    assert(opt.net_args.regression_type in REGRESSION_MAP)
    if opt.net_args.regression_type == 'InnerProductOptimization':
        return InnerProductOptimization(matcher, opt)
            


