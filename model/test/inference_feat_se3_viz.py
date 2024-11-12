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
from tqdm import tqdm
import sys, os, ipdb
from model.vnn.vn_dgcnn import VnDgcnn
from model.option import gen_options
from model.equiv_registration import EquivRegistration
from model.utils import filter_ckpt_state_dict
from model.example_opt import create_opt

#from data_loader.modelnet import ModelNet40Alignment
from data_loader.factory import create_datastream
from model.plot.feat_SE3_vis import visualize_vector_field
import torch
import pdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pypose as pp
import pypose.optim as ppos
import getpass
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
#%matplotlib
import matplotlib.pyplot as plt

def save_to_gif(iters,
                file_ext="png",
                frame_prefix=None,
                file_prefix=None):
    frames = []
    imgs = glob.glob("*."+file_ext)
    for i in range(iters):
        new_frame = Image.open(frame_prefix+str(i)+"."+file_ext)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(file_prefix+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=10, loop=0)



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
    num_files = 2
    #is_using_untrained = True
    ####################
    

    
    ########################
    # Modelnet:
    opt = create_opt(sys.argv[1])
    opt.train_args.num_training_optimization_iters = 400
    #########################
    #        --is-se3 \
    #     --is-centerize \
    #     --is-learning-kernel  \
    #    --is-gradient-ascent \
    #    --min-correlation-threshold 0.01 \
    #    --pose-regression-learning-rate  1e-7 \
    #    --use-full-inner-product \
    #    --use_normalized_pose_loss \
    #    --ell-learning-rate 1e-8 \
    #    --gram-mat-min-nonzero 1000 \
    #    --init-ell 0.2 \
    #    --min-ell 0.025 \
    #    --max-ell 0.25 \

    #db_train = ModelNet40Alignment(opt)
    _, _, testDataLoader = create_datastream(opt)
    
    #dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    #opt.net_args.num_optimization_iters = 1
    #opt.exp_args.batch_size = 1

    encoder = EquivRegistration(opt).cuda()
    checkpoint = torch.load(os.path.join(opt.exp_args.pretrained_model_dir, 'best_model.pth'))
    model_dict = filter_ckpt_state_dict(checkpoint['model_state_dict'], encoder.state_dict(), ['matcher.T', 'regression.matcher.T'])
    encoder.load_state_dict(model_dict, strict=False)
    
    encoder.set_optimization_iters(2)
    for file_index, data in enumerate(testDataLoader):

        pc1 = data['pc1'][:, :, :]
        pc2 = data['pc2'][:, :, :]
        T_gt = data['T']
        if opt.net_args.is_se3:
            t_label = data['t']
        batch_size = pc1.shape[0]
        #if opt.exp_args.is_eval_traj:
        #    seq_name = data['seq']
        #    pc1_id = data['pc1_id']
        #    pc2_id = data['pc2_id']

            
            
        T_init = pp.identity_SE3(1).cuda() if opt.net_args.is_se3 else pp.identity_SO3(1).cuda() 
        T_init.requires_grad = True

            
        #convert to [ batch, 3, num_pts]
        pc1_in = torch.transpose(pc1, 2, 1).cuda()
        pc2_in = torch.transpose(pc2, 2, 1).cuda()
        pc1_in.requires_grad = False
        pc2_in.requires_grad = False
        pc1_np = pc1_in.cpu().detach().numpy()

        for i in tqdm(range(opt.train_args.num_training_optimization_iters)):

            ### pc format from data loader: [batch, num_points, 3]
            ### Vnn input format: [batch, 3, num_points]
            ### Vnn feat format: [batch, channel, 3, num_points]
            pred, feat1, feat2, is_converged = encoder(pc1_in, pc2_in, T_init)
            print("Pred is ",pred)

            pc2_np = (pred[0] @ pc2_in.permute(0,2,1)).permute(0,2,1).cpu().detach().numpy()
            feat1 = feat1.cpu().detach().numpy()
            T_gt_tensor = pp.from_matrix(T_gt, ltype=pp.SE3_type if opt.net_args.is_se3 else pp.SO3_type).cuda()            
            feat2_in_1 = torch.transpose(pred[0] @ torch.transpose(feat2, 3, 2), 3, 2).cpu().detach().numpy()
            feat2_in_1_gt = torch.transpose(T_gt_tensor @ torch.transpose(feat2, 3, 2), 3, 2).cpu().detach().numpy() 
            

            visualize_vector_field(pc1_np, feat1, pc2_np, feat2_in_1,
                                   num_vis_feats=1,
                                   num_vis_points=20, #pc1_in.shape[-1],
                                   file_prefix="iter_"+str(i))

            # pc1 features
            #visualize_vector_field(pc1_np, feat1, None, None,
            #                       num_vis_feats=1,
            #                       num_vis_points=20, #pc1_in.shape[-1],
            #                       file_prefix="pc1_feat_iter_"+str(i))
            # pc1 points
            #visualize_vector_field(pc1_np, None, None, None,
            #                       num_vis_feats=1,
            #                       num_vis_points=20, #pc1_in.shape[-1],
            #                       file_prefix="pc1_iter_"+str(i))
            
            # pc2 feature 
            #visualize_vector_field(None, None, pc2_np, feat2.cpu().detach().numpy(),
            #                       num_vis_feats=1,
            #                       num_vis_points=20,#, pc1_in.shape[-1],
            #                       file_prefix="pc2_feat_iter_"+str(i))
            
            ### pc 2 only
            #visualize_vector_field(None, None, pc2_np, None,
            #                       num_vis_feats=1,
            #                       num_vis_points=20,#, pc1_in.shape[-1],
            #                       file_prefix="pc2_iter_"+str(i))

            # pc1 and pc2 under gt, with features
            #pc2_gt = (T_gt_tensor.cpu() @ pc2).transpose(2,1).cpu().detach().numpy()
            #visualize_vector_field(pc1_np, feat1, pc2_gt, feat2_in_1_gt,
            #                       num_vis_feats=1,
            #                       num_vis_points=20, #pc1_in.shape[-1],
            #                       file_prefix="pc12_gt_iter_"+str(i))
            
            
            
            
            
            #visualize_before_after_2(pc1_np, feat1 , pc2_np, feat2 ,  'iter_'+str(i), is_using_untrained)
            
            T_init = pred
            if  i > opt.train_args.num_training_optimization_iters * 3 / 4:

                encoder.matcher.coord_kernel.ell.data = torch.tensor([0.05])
            print("T_gt is ", pp.from_matrix(T_gt, ltype=pp.SE3_type if opt.net_args.is_se3 else pp.SO3_type), ", current estimate is ",pred, ", ell is ", opt.net_args.init_ell)
            
        save_to_gif(opt.train_args.num_training_optimization_iters,
                    'iter_',
                    str(file_index))

        if file_index == num_files - 1:
            break

    

    

    

    

    

    

    

    
