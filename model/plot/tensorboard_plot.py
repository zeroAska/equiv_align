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
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import pdb
import numpy as np
import pypose as pp
def tensorboard_plot_pointset_pair(tb_writer,
                                   pc1,
                                   pc2,
                                   T_init_f1_to_f2,
                                   T_est_f1_to_f2,
                                   prefix_name,
                                   global_step,
                                   lietype=pp.SE3_type):

    color = torch.Tensor(np.array([[255,0,0],\
                                   [0,153,0],\
                                   [0,0,255]]))

    T_est_f1_to_f2 = pp.LieTensor(T_est_f1_to_f2, ltype=lietype)
    pc2_init_guess = T_init_f1_to_f2 @ pc2
    pc2_transformed = T_est_f1_to_f2 @ pc2
    vertices = torch.cat((pc1, pc2_init_guess, pc2_transformed), 0).unsqueeze(0)

    colors = torch.zeros_like(vertices, dtype=torch.int)
    colors[0, :pc1.shape[0], :] = color[0, :]
    colors[0, pc1.shape[0]:pc1.shape[0]+pc2_init_guess.shape[0], :] = color[1, :]
    colors[0, pc1.shape[0]+pc2_init_guess.shape[0]:, :] = color[2, :]

    tb_writer.add_mesh(prefix_name + ': Point Cloud Before and After transformation', vertices=vertices, colors=colors, global_step=global_step)
    
    
    
    
    


def tensorboard_plot_grad(tb_writer,
                          network,
                          global_step,
                          str_prefix):
    layers= []
    ave_grads = []
    max_grads = []
    min_grads = []
    for n, p in network.named_parameters():
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            #pdb.set_trace()
            #ave_grads.append(p.grad.mean().detach().cpu())
            #max_grads.append(p.grad.abs().max().detach().cpu())
            #min_grads.append(p.grad.abs().min().detach().cpu())
            ave_grads.append(p.mean().detach().cpu())
            max_grads.append(p.abs().max().detach().cpu())
            #min_grads.append(p.abs().min().detach().cpu())
            tb_writer.add_scalars(str_prefix+str(n)+' weight norm', {'mean': ave_grads[-1],
                                                #'min': min_grads[-1],
                                                'max': max_grads[-1] },global_step)

    #pdb.set_trace()
    #if (len(ave_grads) > 0):
