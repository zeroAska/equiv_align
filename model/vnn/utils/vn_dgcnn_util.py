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
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
def knn(x, k):
    ### input shape: x = [batch, feature_dim, num_pts]
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def knn_tensor(x_feat, x_coord, k):
    ### input shape: x = [batch, feature_dim, num_pts]
    inner_feat = -2*torch.matmul(x_feat.transpose(2, 1), x_feat)
    inner_coord = -2*torch.matmul(x_coord.transpose(2, 1), x_coord)
    
    xx_feat = torch.sum(x_feat**2, dim=1, keepdim=True)
    pairwise_distance_feat = -xx_feat - inner_feat - xx_feat.transpose(2, 1)

    xx_coord = torch.sum(x_coord**2, dim=1, keepdim=True)
    pairwise_distance_coord = -xx_coord - inner_coord - xx_coord.transpose(2, 1)
    pairwise_distance = pairwise_distance_coord * pairwise_distance_feat
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def centerize(x: torch.Tensor):
    ### input shape: x = [batch, 3, num_pts]
    x_mean = x.mean(dim=-1).detach()
    return (x - x_mean.unsqueeze(2).expand(x.shape[0],x.shape[1],x.shape[2])).requires_grad_(x.requires_grad), x_mean

def gather_neighbor_feat(x, idx_neighbor, k):
    batch_size = x.shape[0]
    num_points = x.shape[-1]
    idx_base = torch.arange(0, batch_size).cuda().view(-1, 1, 1)* num_points
    idx = idx_neighbor + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3 #num_dims = 1

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature

def get_graph_feature(x, k=20, idx=None, x_coord=None, is_updating_coord=False):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn_tensor(x, x_coord, k)#knn(x_coord, k=k)
            idx_coord = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    if x_coord is not None and is_updating_coord:
        x_coord_neighbor = gather_neighbor_feat(x_coord, idx_coord, k)
        return feature, x_coord_neighbor
    else:
        return feature



def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature
