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
from model.plot.feat_pose_plot import save_to_gif, visualize_before_after_2
from model.utils import copy_new_leaf_tensor, save_color_ply, save_two_pc_transformed
#from model.option import opt
import pdb, ipdb

class InnerProductOptimization(nn.Module):
    def __init__(self,
                 matcher,
                 opt):
                 #trust_region_radius):
        super(InnerProductOptimization, self).__init__()
        self.num_optimization_iters = opt.net_args.num_optimization_iters
        self.is_se3 = opt.net_args.is_se3
        self.trust_region_radius = opt.net_args.trust_region_radius
        self.matcher = matcher
        self.debug_mode = opt.train_args.debug_mode
        self.is_second_order_optimization = opt.net_args.is_second_order_optimization
        self.is_gradient_ascent = opt.net_args.is_gradient_ascent
        self.pose_regression_learning_rate = opt.net_args.pose_regression_learning_rate
        self.ell_learning_rate = opt.net_args.ell_learning_rate 
        self.gram_mat_min_nonzero = opt.net_args.gram_mat_min_nonzero 
        self.predict_non_iterative = opt.net_args.predict_non_iterative
        self.is_visualizing = opt.net_args.is_visualizing
        self.is_holding_pose_fixed = opt.net_args.is_holding_pose_fixed
        self.clip_norm = opt.train_args.clip_norm
        if self.predict_non_iterative:
            self.non_iterative_mlp = nn.Sequential(
                nn.LazyLinear(32),
                nn.ReLU(),
                nn.Linear(32, 4))

        self.viz_hook_per_iter = None
        if (self.is_visualizing):
            self.viz_hook_per_iter = visualize_before_after_2

        if self.debug_mode:            
            print("LM is ",  self.is_second_order_optimization)

        self.use_full_inner_product = opt.net_args.use_full_inner_product
        self.use_normalized_pose_loss = opt.net_args.use_normalized_pose_loss
        self.min_ell = opt.net_args.min_ell
        self.max_ell = opt.net_args.max_ell

    def set_optimization_iters(self, target_iters):
        self.num_optimization_iters = target_iters

    def set_is_holding_pose_fixed(self, is_fixed):

        self.is_holding_pose_fixed = is_fixed

    def forward(self,
                pc1,
                pc2,
                x: torch.Tensor,
                y: torch.Tensor,
                T_init: pp.LieTensor):

        #if not isinstance(T_init, pp.LieTensor):
        batch_size = pc1.shape[0]
        T = copy_new_leaf_tensor(T_init, pp.SE3_type if self.is_se3 else pp.SO3_type)
        T_identity = pp.identity_like(T).cuda().detach()
        #assert(T.is_leaf)                            
        #else:
        #    T = T_init
        #    T_identity = pp.identity_like(T).cuda().detach()
        #    assert(T.is_leaf)                                        

        ### User reweigthed least squares to do the optimization
        if (self.is_second_order_optimization):
            #self.matcher.T.requires_grad = True
            #solver = ppos.solver.Cholesky()
            #strategy = ppos.strategy.TrustRegion(radius=self.trust_region_radius)
            #strategy = pp.optim.strategy.Adaptive(damping=1e-6)
            #self.matcher.set_T_grad_required(True)
            #optimizer = pp.optim.LM(self.matcher)
            optimizer = pp.optim.GN(self.matcher,
                                    #solver=solver,
                                    #strategy=strategy,
                                    #min=1e-6,
                                    #kernel=pp.optim.kernel.Huber(),
                                    vectorize=False)
            #print("In regresssor.forward(), parameters include:")
            #for param in optimizer.param_groups:
            #    print(param)
            scheduler = StopOnPlateau(optimizer, steps=self.num_optimization_iters, patience=3, decreasing=1e-3, verbose=True)
            #optimizer.add_param_groups({'params': T})

        elif self.predict_non_iterative:
            pred = self.non_iterative_mlp(torch.concat((x,y), 1).flatten(start_dim=1))
            return pp.SO3(torch.nn.functional.normalize(pred,dim=1))
            
        else:
            #import pdb; pdb.set_trace()
            assert(T.requires_grad)
            if self.matcher.is_learning_kernel:
                assert(self.matcher.coord_kernel.ell.is_leaf)
                assert(self.matcher.coord_kernel.ell.requires_grad)
                #assert(T.is_leaf)
                params = [{'params': T}, 
                        {'params': self.matcher.coord_kernel.ell, 'lr': self.ell_learning_rate}]
                #if self.clip_norm > 0:
                #    nn.utils.clip_grad_norm_([self.matcher.coord_kernel.ell], self.clip_norm)
            else:
                params = [T]
            if self.debug_mode:
                print("SGD params: ", params)
            optimizer = torch.optim.SGD(params, 
                                        lr=self.pose_regression_learning_rate,
                                        momentum=0.9,
                                        weight_decay=0.9,
                                        maximize=self.is_gradient_ascent)
            #optimizer = torch.optim.Adam(params, 
            #                           lr=self.pose_regression_learning_rate,
            #                            weight_decay=0.9,
            #                            maximize=self.is_gradient_ascent)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=0.1, gamma=0.5)
            
        iters = 0
        is_converged = False
        while True:

            if (self.is_second_order_optimization):
                residual = optimizer.step(input=(pc1, pc2, x,y))
                scheduler.step(residual)
                if (self.debug_mode):
                    print('LM Residual is  {} at iter {}, T is {}'.format(residual, iters, self.matcher.T.data))
                break
                if (scheduler.continual() == False):
                    break
                iters += 1                    
                if (iters > self.num_optimization_iters):
                    is_converged = True
                    break
                
            else:
                if self.is_se3:
                    if self.use_full_inner_product:
                        residual_12, gram_mat_12 = self.matcher(pc1, pc2, x,y,T)
                        residual_1, _            = self.matcher(pc1, pc1, x,x,T_identity)
                        residual_2, _            = self.matcher(pc2, pc2, y,y,T_identity)
                        residual = 2*residual_12 - residual_1 - residual_2
                    else:
                        residual, gram_mat_12 = self.matcher(pc1, pc2, x, y, T) #/ torch.sqrt(self.matcher(pc1, pc1, x, x, T_identity)) / torch.sqrt( self.matcher(pc2, pc2, y, y, T_identity))
                else:
                    #residual, gram_mat_12 = self.matcher(pc1, pc2, x, y, T)
                    if self.use_full_inner_product:
                        residual_12, gram_mat_12 = self.matcher(pc1, pc2, x,y,T)
                        residual_1, _            = self.matcher(pc1, pc1, x,x,T_identity)
                        residual_2, _            = self.matcher(pc2, pc2, y,y,T_identity)
                        residual = 2*residual_12 - residual_1 - residual_2
                    else:
                        residual, gram_mat_12 = self.matcher(pc1, pc2, x, y, T) #/ torch.sqrt(self.matcher(pc1, pc1, x, x, T_identity)) / torch.sqrt( self.matcher(pc2, pc2, y, y, T_identity))

                if self.use_normalized_pose_loss:
                    batch_size = pc1.shape[0]
                    residual = residual / batch_size

                T_last = copy_new_leaf_tensor(T, pp.SE3_type if self.is_se3 else pp.SO3_type)

                nonzero_valid = True
                for b in range(batch_size):
                    if torch.count_nonzero(gram_mat_12[b]) < self.gram_mat_min_nonzero:
                        print("T_init: ", T)
                        save_two_pc_transformed(pc1[b,:,:].transpose(1,0), None,
                                                pc2[b,:,:].transpose(1,0), None,
                                                T[b,:].matrix(), 'regressor_non_converge_')
                        save_color_ply(pc1[b,:,:].transpose(1,0), None, "regressor_non_converge_pc1.ply")
                        save_color_ply(pc2[b,:,:].transpose(1,0), None, "regressor_non_converge_pc2.ply")
                        print("gram mat too sparse. break the optimization")
                        nonzero_valid = False
                        break
                if nonzero_valid == False:
                    optimizer.zero_grad()
                    is_converged = False
                    break
                
                if (self.debug_mode):
                    print('SGD Residual is  {} at iter {}, T is {}, ell is {}'.format(residual, iters, T, self.matcher.coord_kernel.ell.item()))
                
                residual.backward(retain_graph=True)
                if self.is_holding_pose_fixed == False:
                    optimizer.step()
                if self.matcher.is_learning_kernel:
                    self.matcher.coord_kernel.ell.data.clamp_(min=self.min_ell, max=self.max_ell)
                
                iters += 1                
                if (iters >= self.num_optimization_iters):
                    is_converged = True
                    break
                optimizer.zero_grad()

                if (T_last.Inv() @ T).Log().norm(dim=-1).sum() < 1e-6:
                    print("Change of T less than 1e-6, break at iter {}, ell={}".format(iters, self.matcher.coord_kernel.ell.item()))
                    is_converged = True                    
                    break
                
            if self.viz_hook_per_iter is not None:
                viz_hook_per_iter(pc1, pc2, x, y, T)

        return T , is_converged




