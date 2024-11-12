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
import torch
import pdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pypose as pp
import pypose.optim as ppos

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D, proj3d
matplotlib.use("Agg")
#%matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    ### source: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c    
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)



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

def viz_feat_and_point_with_pose(pc1, feat1, pc2, feat2, T,
                                 file_prefix, is_using_untrained):
    
    pc1_np = pc1.cpu().detach().numpy()    
    feat1 = feat1.cpu().detach().numpy() # shape [(batch, channel, 3, num_pt) ]
    pc2_np = (T @ pc2_in.permute(0,2,1)).permute(0,2,1).cpu().detach().numpy()
    feat2 = torch.transpose(T @ torch.transpose(feat2, 3, 2), 3, 2).cpu().detach().numpy()
    visualize_before_after_2(pc1_np, feat1 , pc2_np, feat2 ,  'iter_'+str(i), is_using_untrained)
    
    #save_to_gif(iters,'iter_', str(file_index))
    
        
    
