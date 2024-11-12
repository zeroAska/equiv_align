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


from model.vnn.vn_dgcnn import VnDgcnn
from model.option import gen_options
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
matplotlib.use("TkAgg")
#%matplotlib
import matplotlib.pyplot as plt

def visualize_before_after(pc1, feat1, pc2, feat2, feat_rotated_directed):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(15, 12))
    
    #===============
    # PC subplot
    #===============

    title_prefix = "Before rotation:"
    ax = fig.add_subplot(2, 2, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="input point clouds, unrotated")
    #X = pc2[:, 0]
    #Y = pc2[:, 1]
    #Z = pc2[:, 2]
    #pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')

    title_prefix = "After rotation:"
    ax = fig.add_subplot(2, 2, 3, projection=Axes3D.name)#projection='3d')
    ax.view_init(60,60)
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')  
    title_prefix = "Before rotation:"    
    X = feat1[0, 0, :]
    Y = feat1[0, 1, :]
    Z = feat1[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="dodgerblue", label="features, unrotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')

    ax = fig.add_subplot(2, 2, 4, projection='3d')  
    title_prefix = "After rotation:"    
    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')


    X = feat3[0, 0, :]
    Y = feat3[0, 1, :]
    Z = feat3[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="g", alpha=0.3, label="feature map directly beging rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_title(title_prefix + ': Features from Equiv Network')

    ax.legend()

    
    plt.show()



def visualize_before_after_2(pc1, feat1, pc2, feat2):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(20, 8))
    
    #===============
    # PC subplot
    #===============
    title_prefix = "Before and After rotation:"
    ax = fig.add_subplot(1, 2, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    #ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="input point clouds, unrotated")
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    title_prefix = "After rotation:"    
    X = feat1[0, 0, :]
    Y = feat1[0, 1, :]
    Z = feat1[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )

    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="g", alpha=0.3, label="feature map directly beging rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network distance is '+str(np.linalg.norm(feat1-feat2) ))    

    ax.legend()
    #plt.show()
    plt.savefig("1.png")


def visualize_before_after_3(pc1, feat1, pc2, feat2, feat3):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(18, 5))
    
    #===============
    # PC subplot
    #===============
    title_prefix = "Before and After rotation: "
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    #ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="input point clouds, unrotated")
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    ax = fig.add_subplot(1, 3, 2, projection='3d')  
    title_prefix = "Features before and After rotation"    
    X = feat1[0, 0, :]
    Y = feat1[0, 1, :]
    Z = feat1[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="dodgerblue", alpha=0.3, label="features, unrotated")
    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz2 = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ', Feature Distance Norm: '+str( np.linalg.norm(feat1-feat2) ))
    ax.legend()

    
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 3, 3, projection='3d')  
    title_prefix = "Features after rotation"    
    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ', Feature Distance Norm: '+str( np.linalg.norm(feat2-feat3) ))

    X = feat3[0, 0, :]
    Y = feat3[0, 1, :]
    Z = feat3[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="g", alpha=0.3, label="feature map directly beging rotated")
    #ax.set_title(title_prefix + ': Features from Equiv Network')

    ax.legend()


    
    plt.show()        
    
    

def visualize_pcs(pc : np.ndarray,
                  pc_feat : np.ndarray,
                  title_prefix : str,
                  fig,
                  row_id):

    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(row_id, 2, 1, projection='3d')
    X = pc[:, 0]
    Y = pc[:, 1]
    Z = pc[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    #===============
    # Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(row_id, 2, 2, projection='3d')  
  
    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    X = pc_feat[0, 0, :]
    Y = pc_feat[0, 1, :]
    Z = pc_feat[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')



if __name__ == '__main__':
    data = ModelNetDataLoader('data/modelnet/EvenAlignedModelNet40PC/',split='test', uniform=False, normal_channel=False,npoint=1024)
    dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    pc = next(iter(dl))
    pc = next(iter(dl)) 
    
    encoder = VnDgcnn(20).cuda()
    checkpoint = torch.load( 'checkpoints/best_model.pth')
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

    ### pc format: [batch, num_points, 3]
    ### Vnn input format: [batch, 3, num_points]
    ### Vnn feat format: [batch, channel, 3, num_points]
    x_in = torch.transpose(pc[0], 2, 1).cuda()

    pdb.set_trace()
    feat, _ = encoder(x_in)

    pc1 = x_in.cpu().detach().numpy()
    feat1 = feat.cpu().detach().numpy()
    #visualize_pcs(x_in.cpu().detach().numpy(),
    #              feat.cpu().detach().numpy(),
    #              "Before rotation")

    T = pp.SO3([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
    #pdb.set_trace()            
    pc_rotated = T @ pc[0]
    x_in = torch.transpose(pc_rotated, 2, 1).cuda()
    feat_rotated, _ = encoder(x_in)
    pc2 = x_in.cpu().detach().numpy()
    feat2 = feat_rotated.cpu().detach().numpy()

    #pdb.set_trace()
    feat3 = torch.transpose(T @ torch.transpose(feat.cpu(), 3, 2), 3, 2).cpu().detach().numpy()
    
    #visualize_pcs(x_in.cpu().detach().numpy(),
    #              feat_rotated.cpu().detach().numpy(),
    #              "After rotation")
    #visualize_before_after_3(pc1, feat1, pc2, feat2, feat3)
    pdb.set_trace()
    visualize_before_after_2(pc1, feat1, pc2, feat2)
    

    

    

    

    

    

    

    
