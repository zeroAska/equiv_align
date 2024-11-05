import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppos
#from option import opt
from .iterative import IterativeMatch
from .single_kernel_matcher import SingleKernelMatch

#class EquivMatcher(nn.Module):

MATCHER_MAP = {
    'IterativeMatch': IterativeMatch,

    'SingleKernelMatch': SingleKernelMatch
}
    
#    def __init__(self,
#                 opt):


#super(EquivMatcher, self).__init__()

### register str to encoder
def gen_matcher(opt):
    assert(opt.net_args.matcher_type in MATCHER_MAP)
    if opt.net_args.matcher_type == 'IterativeMatch':
        return IterativeMatch(opt)
    elif opt.net_args.matcher_type == 'SingleKernelMatch':
        return SingleKernelMatch(opt)
    else:
        raise Exception("Unimplemented matcher type {}".format(opt.net_args.matcher_type))

        
            
