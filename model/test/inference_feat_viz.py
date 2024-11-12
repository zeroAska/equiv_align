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

from PIL import Image
import glob
import sys, os
from model.vnn.vn_dgcnn import VnDgcnn
from model.option import gen_options
from model.equiv_registration import EquivRegistration
from data_loader.modelnet import ModelNetDataLoader
import torch
import pdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pypose as pp
import pypose.optim as ppos

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
#%matplotlib
import matplotlib.pyplot as plt

def save_to_gif(iters,
                frame_prefix=None,
                file_prefix=None):
    frames = []
    imgs = glob.glob("*.png")
    for i in range(iters):
        new_frame = Image.open(frame_prefix+str(i)+".png")
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(file_prefix+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)


### pc1 input format: [batch, 3, num_points]
def visualize_before_after_2(pc1, feat1, pc2, feat2, file_prefix, is_using_untrained):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(20, 8))
    
    #===============
    # PC subplot
    #===============
    title_prefix = file_prefix + ": Source and Target:"
    ax = fig.add_subplot(1, 2, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    #ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="Source input point clouds")
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="Target input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_title(title_prefix + ' Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    title_prefix = file_prefix + ": Source and Target:"    
    X = feat1[0, 0, 0, :]
    Y = feat1[0, 0, 1, :]
    Z = feat1[0, 0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="dodgerblue", alpha=0.3, label="Source features" )

    X = feat2[0, 0, 0, :]
    Y = feat2[0, 0, 1, :]
    Z = feat2[0, 0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", alpha=0.3, label="Target features")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if is_using_untrained:
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
        ax.set_zlim(-4,4)
    else:
        ax.set_xlim(-1e-9,1e-9)
        ax.set_ylim(-1e-9,1e-9)
        ax.set_zlim(-1e-9,1e-9)
    
    ax.set_title(title_prefix + ' Features from Equiv Network distance is '+str(np.linalg.norm(feat1-feat2) ))    

    ax.legend()
    #plt.show()
    plt.savefig(file_prefix + ".png")



if __name__ == '__main__':
    ####################
    ##  params
    num_files = 5
    iters = 100
    is_using_untrained = True
    ####################
    
    opt = gen_options()
    data = ModelNetDataLoader('/home/rayzrzh/code/data/modelnet/modelnet40_normal_resampled/',split='test', uniform=False, normal_channel=False,npoint=1024)
    dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    opt.net_args.num_optimization_iters = 1
    opt.exp_args.batch_size = 1
    dl = iter(dl)

    encoder = EquivRegistration(opt).cuda()    
    if is_using_untrained:
        ckpt_path = os.path.join(opt.exp_args.pretrained_model_dir, 'best_model.pth')
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load( 'log/2023-07-18_19-15-train_multi_backprop_large_scale/checkpoints/best_model.pth')
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    

    for file_index in range(num_files):
        pc = next(dl)
        #pc = next(iter(dl))

        T = pp.SO3([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
        T_init = pp.identity_SO3(1).cuda()
        T_init.requires_grad = True
        pc2 = T @ pc[0]
        #pdb.set_trace()

        #convert to [ batch, 3, num_pts]
        pc1_in = torch.transpose(pc[0], 2, 1).cuda()
        pc2_in = torch.transpose(pc2, 2, 1).cuda()
        pc1_in.requires_grad = False
        pc2_in.requires_grad = False

        pc1_np = pc1_in.cpu().detach().numpy()    




        for i in range(iters):

            ### pc format from data loader: [batch, num_points, 3]
            ### Vnn input format: [batch, 3, num_points]
            ### Vnn feat format: [batch, channel, 3, num_points]
            #pdb.set_trace()

            pred, feat1, feat2 = encoder(pc1_in, pc2_in, T_init)


            feat1 = feat1.cpu().detach().numpy() # shape [(batch, channel, 3, num_pt) ]
            #feat2 = feat2.cpu().detach().numpy()
            pc2_np = (T_init[0] @ pc2_in.permute(0,2,1)).permute(0,2,1).cpu().detach().numpy()
            feat2 = torch.transpose(T_init[0] @ torch.transpose(feat2, 3, 2), 3, 2).cpu().detach().numpy() 
        
            visualize_before_after_2(pc1_np, feat1 , pc2_np, feat2 ,  'iter_'+str(i), is_using_untrained)

        save_to_gif(iters,'iter_', str(file_index))

    

    

    

    

    

    

    

    
