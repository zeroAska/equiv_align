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
            


