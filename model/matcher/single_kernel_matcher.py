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
import numpy
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
#from model import option.opt as opt
import pypose as pp
import pypose.optim as ppos
from matplotlib import pyplot as plt
# kernel 
import pdb
import ipdb
from .kernel import gen_kernel
#from ..option import opt

class SingleKernelMatch(nn.Module):
    def __init__(self,
                 opt):
                 #kernel_module):

        super(SingleKernelMatch, self).__init__()
        self.is_learning_kernel = opt.net_args.is_learning_kernel
        #self.kernel = gen_kernel(opt.net_args.kernel_type, opt.net_args.init_ell,
        #                         opt.net_args.is_learning_kernel)
        self.coord_kernel = gen_kernel(opt.net_args.kernel_type, opt.net_args.init_ell, opt.net_args.is_learning_kernel)
        
        self.min_correlation_threshold = opt.net_args.min_correlation_threshold
        self.debug_mode = opt.train_args.debug_mode

        if (self.debug_mode):
            for (name, param) in self.named_parameters():
                print("params: ", name, param)
            #for (name, param) in self.kernel.named_parameters():
            #    print("params: ", name, param)
        self.is_visualizing = opt.net_args.is_visualizing
        self.is_output_gram_mat = opt.net_args.predict_non_iterative
        #print(self.parameters())
        

    def set_T_grad_required(self, requires_grad:bool):
        #if (self.debug_mode):
        #    print("Current T grad requires is ", requires_grad)
        self.T.requires_grad = requires_grad

    def visualize_correlation(self, pc1, pc2, feat1, feat2, correlation):

        fig = plt.figure(figsize=(18, 5))
        
        axs0 = fig.add_subplot(1, 3, 1, projection='3d')
        pc1 = pc1[0, :, :].detach().cpu()
        pc2 = pc2[0, :, :].detach().cpu()
        #if (self.debug_mode):
        print("pc1 shape: ", pc1.shape)
        print("pc2 shape: ", pc2.shape)
        axs0.scatter(pc1[ 0, :], pc1[1,:], pc1[2,:], marker='o', color="blue", label="pc1")
        axs0.scatter(pc2[0, :], pc2[1,:], pc2[2,:], marker='o', color="red", label="pc2")
        axs0.set_title("Two input point clouds's raw 3D coordinates")
        #axs[0].legend()

        axs0 = fig.add_subplot(1, 3, 2, projection='3d')
        feat1 = feat1[0, 0, :, :].detach().cpu()
        feat2 = feat2[0, 0, :, :].detach().cpu()
        print("feat1 shape: ", feat1.shape)
        print("feat2 shape: ", feat2.shape)
        axs0.scatter(feat1[ 0, :], feat1[1,:], feat1[2,:], marker='o', color="blue", label="feature 1")
        axs0.scatter(feat2[0, :], feat2[1,:], feat2[2,:], marker='o', color="red", label="feature 2")
        axs0.set_title("Two input point clouds's feature maps")
        

        axs1 = fig.add_subplot(1, 3, 3)        
        img1 = axs1.imshow(correlation.detach().cpu()[0,:,:], interpolation='none', cmap='Greys')
        #cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        axs1.set_xlabel('pc1 superpixel feature map index')
        axs1.set_ylabel('pc2 superpixel feature map index')
        axs1.set_title('dense pairwise correlation value. \nNumber of Matches: '+ str( torch.count_nonzero( correlation).detach().cpu().numpy()) +' out of '+ str(pc1.shape[1] * pc2.shape[1]))
        fig.colorbar(img1)#cax=cax)
        axs1.legend()
        plt.show()

    def set_init_T(self, init_T: pp.LieTensor):
        if isinstance(init_T, pp.LieTensor):
            T_pp = init_T
        else:
            T_pp = pp.LieTensor(init_T, ltype=pp.SO3_type)
        
        #if (self.debug_mode):
        #    print("Change init T from {} to {}".format(self.T, T_pp))
        self.T = pp.Parameter(T_pp)#, requires_grad=True)
        #self.T_dim = init_T.ltyle.dimension[0]


    def initialize_kernel(self):
        #self.kernel.initialize_kernel()
        self.coord_kernel.initialize_kernel()


 
    '''
    x, y are the transformed feature maps of the two input point clouds
    x, y dimension: [batch, channel, rot_dim, num_points ]
    Each point's feature map is c x rot_dim
    
    '''
    def forward(self, pc1, pc2,
                x_feat, y_feat, #,):
                T_tensor: pp.LieTensor):

        #self.T.data = 
        #self.T = nn.Parameter(T_tensor.cuda())#.clone().data #nn.Parameter(T_tensor)

        #T_init 
        ### permute the tensor so that T can act on the last dimension
        ###   x, y dimension: [batch, num_points, channel, rot_dim]
        batch_size = pc1.shape[0]
        x = torch.permute(x_feat, (0, 3, 1, 2))
        y = torch.permute(y_feat, (0, 3, 1, 2))

        pc1 = torch.permute(pc1, (0, 2,1))
        pc2 = torch.permute(pc2, (0, 2,1))
        
        x = torch.concat((pc1.unsqueeze(2),
                          x), dim = 2)
        y = torch.concat((pc2.unsqueeze(2),
                          y), dim = 2)
        

        ### group action
        #T_batch = self.T.unsqueeze(1).unsqueeze(1).expand(x.shape[0],
        #if not isinstance(T_tensor, pp.LieTensor):
        #    T_tensor = pp.SO3(T_tensor)
        
        T_batch = T_tensor.unsqueeze(1).unsqueeze(1).expand(x.shape[0],
                                                            x.shape[1],
                                                            x.shape[2],
                                                            T_tensor.shape[-1])
                                                            #self.T.shape[-1])

        y = T_batch.rotation() @ y
        
        #print("T_batch after being expanded is {} with shape {}".format(T_batch, T_batch.shape))        
        ### flatten to [batch, num_points, flattned_feature_vec]
        x_flat = torch.flatten(x, start_dim=2, end_dim=3)
        y_flat = torch.flatten(y, start_dim=2, end_dim=3)

        ### pairwise dist gram matrix for these superpixels, normalized
        gram_mat = self.coord_kernel(x_flat, y_flat) #/ x_flat.shape[1] / y_flat.shape[1]
            
 
        if (self.min_correlation_threshold > 0):
            gram_mat = torch.where((gram_mat > self.min_correlation_threshold),
                                   gram_mat, 0.0)
        #x    gram_mat = gram_mat[gram_mat < self.min_correlation_threshold]
        if self.debug_mode:
            for b in range(batch_size):
                print("Nonzero terms for batch index {} is {}".format(b, torch.count_nonzero(gram_mat[b])))

        #if self.debug_mode and self.is_visualizing:
        #    self.visualize_correlation(pc1, pc2, x_feat, y_feat, gram_mat)
        
        #ip_sum = -torch.sum(gram_mat, dim=(1,2))
        ip_sum = torch.sum(gram_mat)


        #if self.debug_mode:
        #    print("inner product shape is {}".format(ip_sum.shape))
        #return distance, gram_mat, ip_sum, weighted_inner_product #self.kernel.forward(distance)
        #if self.is_output_gram_mat:
        return ip_sum, gram_mat
        #else:
        #    return ip_sum #self.kernel.forward(distance)
        
        

        
        
        
