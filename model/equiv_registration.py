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
import torch.nn as nn
import pypose as pp
import pypose.optim as ppos
from .encoder import EquivEncoder
from .matcher.matcher import gen_matcher
from .utils import save_color_ply, save_two_pc_transformed
import ipdb
from .regression.regression import gen_regression
#from option import opt
from .preprocess import Preprocessor
from torch.utils.tensorboard import SummaryWriter
from model.utils import copy_new_leaf_tensor

import ipdb

class EquivRegistration(nn.Module):
    def __init__(self,
                 opt):
        super(EquivRegistration, self).__init__()
        self.is_train = (opt.exp_args.run_mode == 'train')
        self.encoder = EquivEncoder(opt)
        self.matcher = gen_matcher(opt)
        self.regression = gen_regression(self.matcher, opt)
        self.use_feature_pyramid = opt.net_args.use_feature_pyramid
        self.is_se3 = opt.net_args.is_se3
        self.debug_mode = opt.train_args.debug_mode
        self.is_updating_coord = opt.net_args.is_updating_coord
        self.preprocessor = Preprocessor(opt.net_args.is_centerize,
                                         opt.net_args.is_normalize)


        if self.is_train == False:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.debug_mode:
            self.counter = 0
            for n, p in self.named_parameters():
                print("{} with requires_grad={}".format(n, p.requires_grad))

    def get_trainable_params(self):
        ret_encoder  = [ m for m in self.encoder.parameters()]
        ret_decoder = [m for m in self.regression.parameters()]
        return ret_encoder + ret_decoder
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.LazyLinear, nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.04)

    def set_optimization_iters(self, target_iters):
        self.regression.set_optimization_iters(target_iters)



    def dry_run(self, batch: int, num_pts: int, tensorboard_writer: SummaryWriter) -> None:

        #self.matcher.set_init_T(pp.identity_SO3(batch).cuda())
        optimization_iters = self.regression.num_optimization_iters
        self.regression.set_optimization_iters(1)
        if self.is_se3:
            T_init = pp.identity_SE3(batch).cuda()
        else:
            T_init = pp.identity_SO3(batch).cuda()
        T_init.requires_grad = True
        pc1 =10 * torch.rand(batch , 3, num_pts ).cuda()#.requires_grad_(True)
        pc2 =-10 * torch.rand(batch , 3, num_pts ).cuda()#.requires_grad_(True)
        
        self.forward(pc1, pc2, T_init)

        #if tensorboard_writer is not None:
        #    tensorboard_writer.add_graph(self, (pc1, pc1, T_init)) 

        self.regression.set_optimization_iters(optimization_iters)
        return

    def forward(self,
                pc_x: torch.Tensor, # shape: b, 3, num_pts
                pc_y: torch.Tensor,
                T_init: pp.LieTensor):

        self.matcher.initialize_kernel()
        #T_init = copy_new_leaf_tensor(T_init, pp.SE3_type if self.is_se3 else pp.SO3_type)

        x_centered, y_centered, t_shift_x, t_shift_y, normalize_scale = self.preprocessor(pc_x, pc_y)
        if self.regression.is_holding_pose_fixed:
            T_init_rotation = pp.SO3(T_init[:, -4:], ltype=pp.SO3_type) .detach()
            T_init_translation = T_init[:, :3].detach()
            t_shift = (T_init_rotation @ t_shift_y) + T_init_translation -  t_shift_x
            T_init = pp.SE3(torch.cat((t_shift, T_init_rotation), dim=-1))
          
        if False:
        #if True and self.regression.is_holding_pose_fixed:
            print("T_init: ", T_init)
            save_color_ply(torch.cat((x_centered[0,:,:], y_centered[0,:,:]), dim=-1).transpose(1,0).cpu().numpy(), None, "encoder_init_stacked.ply")
            save_two_pc_transformed(x_centered[0,:,:].transpose(1,0), None,
                                    y_centered[0,:,:].transpose(1,0), None,
                                    T_init[0,:].matrix(), 'encoder_')
            save_color_ply(x_centered[0,:,:].transpose(1,0), None, "encoder_pc1.ply")
            save_color_ply(y_centered[0,:,:].transpose(1,0), None, "encoder_pc2.ply")

        equiv_feat_x, feat_pyramid_x, x_coord = self.encoder(x_centered)
        equiv_feat_y, feat_pyramid_y, y_coord = self.encoder(y_centered)
        T_result, is_converged = self.regression(x_coord, y_coord,
                                                 equiv_feat_x,
                                                 equiv_feat_y,
                                                 T_init)
        if False and is_converged:
        #if True and self.regression.is_holding_pose_fixed:
            print("T_init: ", T_init)
            print("T_result: ", T_result)            
            save_color_ply(torch.cat((x_centered[0,:,:], y_centered[0,:,:]), dim=-1).transpose(1,0).cpu().numpy(), None, "regressor_init_stacked.ply")
            save_two_pc_transformed(x_centered[0,:,:].transpose(1,0), None,
                                    y_centered[0,:,:].transpose(1,0), None,
                                    T_result[0,:].matrix(), 'regressor_')
            save_color_ply(x_centered[0,:,:].transpose(1,0), None, "regressor_pc1.ply")
            save_color_ply(y_centered[0,:,:].transpose(1,0), None, "regressor_pc2.ply")
        
        
        if self.is_se3 and is_converged:
            batch_size = pc_x.shape[0]
            t_shift = - (T_result.rotation() @ t_shift_y) + torch.mul(normalize_scale.unsqueeze(1).expand(batch_size, 3),  
                                                                      T_result.translation()) + t_shift_x
            T_result = pp.SE3(torch.cat((t_shift, T_result.rotation()), dim=-1))

        if False and self.is_se3 and is_converged:
        #if True and self.regression.is_holding_pose_fixed:
            print("T_init: ", T_init)
            print("T_result_non_centered: ", T_result)            
            save_color_ply(torch.cat((pc_x[0,:,:], pc_y[0,:,:]), dim=-1).transpose(1,0).cpu().numpy(), None, "result_{}_init_stacked.ply".format(self.counter))
            save_two_pc_transformed(pc_x[0,:,:].transpose(1,0), None,
                                    pc_y[0,:,:].transpose(1,0), None,
                                    T_result[0,:].matrix(), 'result_{}_'.format(self.counter))
            save_color_ply(pc_x[0,:,:].transpose(1,0), None, "result_{}_pc1.ply".format(self.counter))
            save_color_ply(pc_y[0,:,:].transpose(1,0), None, "result_{}_pc2.ply".format(self.counter))
            self.counter += 1
            

        is_converged = torch.Tensor([is_converged]).cuda()
        if self.use_feature_pyramid:
            return T_result, feat_pyramid_x, feat_pyramid_y, is_converged
        elif self.is_updating_coord:
            return T_result, x_coord, y_coord, is_converged
        else:
            return T_result, equiv_feat_x, equiv_feat_y, is_converged
    
class LossModule(torch.nn.Module):
    def __init__(self, opt):
        super(LossModule, self).__init__()
        self.ltype = pp.SE3_type if opt.net_args.is_se3 else pp.SO3_type
        if opt.train_args.loss_type == "matrix_logarithm":
            self.loss_f = lambda x,y: torch.norm(pp.Log((x.Inv() @ y)), p=2)
        else:
            self.loss_f = lambda x,y: torch.norm((x.Inv() @ y).matrix() - pp.identity_like(x).cuda().matrix() )

    def forward(self, pred, target):
        ''' Calculate pose training loss for the encoder '''

        #target = target.contiguous()#.view(-1)
        if not isinstance(pred, pp.LieTensor): #or \
            pred = pp.LieTensor(pred, ltype=self.ltype)
        if not isinstance(target, pp.LieTensor):        
            target = pp.LieTensor(target, ltype=self.ltype)
        
        #ipdb.set_trace()
        loss = self.loss_f(pred, target)
            
        return loss 



        #for iter in range(self.num_optimization_iters):
            
        
        
    
