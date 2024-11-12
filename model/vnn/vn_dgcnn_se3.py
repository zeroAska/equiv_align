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
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.vnn.vn_layers import *
from model.vnn.utils.vn_dgcnn_util import get_graph_feature, centerize
import pdb

class VnDgcnnSE3(nn.Module):
    def __init__(self,
                 n_knn,
                 normal_channel=False,
                 pooling='max',
                 is_centerize=False):
        super(VnDgcnnSE3, self).__init__()
        #self.args = args
        #self.n_knn = args.n_knn
        self.n_knn = n_knn
        self.is_centerize = is_centerize
        #self.num_class = 40
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
            self.pool5 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    ### input shape: x = [batch, 3, num_pts]
    def forward(self, x):
        coord_x = x
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn, x_coord=coord_x)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn, x_coord=coord_x)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn, x_coord=coord_x)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn, x_coord=coord_x)
        x = self.conv4(x)
        x4 = self.pool4(x) # format: [batch, channel, 3, num_pts]
        
        #x5 = mean_pool(x4)
        x_pyramid = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_pyramid)
        #print("x shape:", x.shape)
        #print("x4 shape: ",x4.shape)
        #print("x5 shape: ",x5.shape)
        
        #assert(x_input.requires_grad is True)
        #assert(x4.requires_grad is True)
        #if x_translation is not None:
        #    assert(x_translation.requires_grad is True)
        '''
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        trans_feat = None
        '''
        return x4, x_pyramid, coord_x

