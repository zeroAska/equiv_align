import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppos
#from option import opt
from .vnn.vn_dgcnn import VnDgcnn
from .vnn.vn_dgcnn_se3 import VnDgcnnSE3
from .raw_encoder import IdentityEncoder

class EquivEncoder(nn.Module):

    ENCODER_MAP = {
        'VnDgcnn': VnDgcnn,
        'VnDgcnnSE3': VnDgcnnSE3,
        'IdentityEncoder': IdentityEncoder
    }
    
    def __init__(self,
                 opt):

        super(EquivEncoder, self).__init__()

        ### register str to encoder
        assert(opt.net_args.encoder_type in self.ENCODER_MAP, opt.net_args.encoder_type)
        if (opt.net_args.encoder_type == 'VnDgcnn'):
            self.encoder = VnDgcnn(opt.net_args.n_knn,
                                   is_centerize = opt.net_args.is_centerize)
        elif opt.net_args.encoder_type == 'VnDgcnnSE3':
            self.encoder = VnDgcnnSE3(opt.net_args.n_knn,
                                   is_centerize = opt.net_args.is_centerize)
        elif opt.net_args.encoder_type == 'IdentityEncoder':
            self.encoder = IdentityEncoder()
        else:
            raise("Runtime error: encoder not implemented")

    def forward(self, x):
        return self.encoder(x)
        
            

