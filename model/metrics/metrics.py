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
import numpy as np
import os
import torch
import pdb
import pypose as pp
import pypose.optim as ppos


def pose_log_norm(pred: pp.LieTensor,
                      target: pp.LieTensor,
                      ltype=pp.SO3_type):
    if not isinstance(pred, pp.LieTensor):
        pred = pp.LieTensor(pred, ltype=ltype)
    if not isinstance(target, pp.LieTensor):        
        target = pp.LieTensor(target, ltype=ltype)
        
    err = list( (pred.Inv() @ target).Log().norm(dim=-1).detach() )

    return err

def pose_Fro_norm(pred: pp.LieTensor,
                  target: pp.LieTensor,
                  ltype=pp.SO3_type):
    if not isinstance(pred, pp.LieTensor):
        pred = pp.LieTensor(pred, ltype=ltype)

    if not isinstance(target, pp.LieTensor):        
        target = pp.LieTensor(target, ltype=ltype)

    return list( ((pred.Inv() @ target).matrix() - pp.identity_like(pred).matrix() ).norm(dim=(-2,-1)).detach()  )
    #return list( (pred.Inv() @ target).Log().norm(dim=1).detach().numpy() )

def translation_error(pred: pp.LieTensor,
                      target: pp.LieTensor,
                      ltype=pp.SO3_type):
    
    if not isinstance(pred, pp.LieTensor):
        pred = pp.LieTensor(pred, ltype=ltype)

    if not isinstance(target, pp.LieTensor):        
        target = pp.LieTensor(target, ltype=ltype)

    #assert(pred.is_SE3() and target.is_SE3())

    err = list((pred.translation() - target.translation()).norm(dim=-1).detach())
    
    return err
    
    
def rotation_angle_error(pred: pp.LieTensor,
                         target: pp.LieTensor,
                         ltype=pp.SO3_type):
    
    if not isinstance(pred, pp.LieTensor):
        pred = pp.LieTensor(pred, ltype=ltype)

    if not isinstance(target, pp.LieTensor):        
        target = pp.LieTensor(target, ltype=ltype)

    diff = pred.Inv() @ target
    angle_err = list( diff.euler().norm(dim=-1)/ np.pi * 180 )
    return angle_err


def map_err_ops_to_list(pred, target,
                        op_list,
                        name_list):
    for op in op_list:
        name_list.append(op(pred, target))
