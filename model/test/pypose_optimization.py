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
import torch
from torch import nn
import pypose as pp
import pypose.optim as ppos
from pypose.optim.scheduler import StopOnPlateau

class PoseInv(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_se3(*dim))

    def forward(self, input):
        # the last dimension of the output is 6,
        # which will be the residual dimension.
        return (self.pose.Exp() @ input).Log()

posinv = PoseInv(2, 2)
input = pp.randn_SE3(2, 2)
strategy = pp.optim.strategy.Adaptive(damping=1e-6)
optimizer = pp.optim.LM(posinv, strategy=strategy)

for idx in range(10):
    loss = optimizer.step(input)
    print('Pose Inversion loss %.7f @ %d it'%(loss, idx))
    if loss < 1e-5:
        print('Early Stopping with loss:', loss.item())
        break
