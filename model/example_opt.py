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
from model.option import gen_options
import getpass
def create_opt(dataset_name, add_noise=False):
    opt = gen_options()

    if dataset_name == 'ModelNet40Alignment':
        ########################
        # Modelnet:
        opt.exp_args.dataset_name = 'ModelNet40Alignment' 
        opt.exp_args.dataset_path = "/home/"+getpass.getuser()+"/data/modelnet/EvenAlignedModelNet40PC"
        opt.exp_args.pretrained_model_dir = "log/modelnet_airplane_90/checkpoints"
        opt.exp_args.modelnet_airplane_only = True    

 
        opt.exp_args.gpus = "0"
        opt.exp_args.noise_augmentation = 0.1 if add_noise else 0.0
        opt.exp_args.outlier_augmentation = 0.2 if add_noise else 0.0
        opt.exp_args.max_rotation_degree = 90.0
        opt.exp_args.max_translation_norm = 0.0
        opt.net_args.pose_regression_learning_rate = 1e-5
        opt.exp_args.batch_size = 1
        opt.exp_args.is_shuffle = False
        opt.exp_args.skip_random_permutation = True
        opt.net_args.ell_learning_rate = 1e-6
        opt.net_args.num_optimization_iters = 800
        opt.net_args.is_learning_kernel = True
        opt.net_args.is_gradient_ascent = True
        opt.net_args.gram_mat_min_nonzero = 1000
        opt.net_args.use_full_inner_product = True
        opt.net_args.use_normalized_pose_loss = True
        opt.net_args.min_ell = 0.05
        opt.net_args.init_ell = 0.25
        opt.net_args.max_ell = 1.0
        opt.net_args.is_se3 = False
        opt.net_args.is_centerize = False
        opt.train_args.num_training_optimization_iters = 800
        opt.train_args.debug_mode = True

        #db_train = ModelNet40Alignment(opt)
        
    elif dataset_name == 'eth3d':

        ########################
        # Modelnet:
        opt.exp_args.dataset_name = 'eth3d' 
        opt.exp_args.dataset_path = "/home/"+getpass.getuser()+"/data/eth3d/"
        opt.exp_args.pretrained_model_dir = "log/eth3d_largescale/checkpoints"
        opt.exp_args.is_auto_split_dataset = True
        opt.exp_args.gpus = "0"
        opt.exp_args.max_rotation_degree = 10.0
        opt.net_args.pose_regression_learning_rate = 1e-7
        opt.exp_args.batch_size = 1
        opt.exp_args.num_workers = 0
        opt.exp_args.is_shuffle = False
        opt.net_args.num_optimization_iters = 400        
        opt.net_args.ell_learning_rate = 1e-8
        opt.net_args.is_learning_kernel = True
        opt.net_args.is_gradient_ascent = True
        opt.net_args.gram_mat_min_nonzero = 1000
        opt.net_args.use_full_inner_product = True
        opt.net_args.use_normalized_pose_loss = True
        opt.net_args.min_ell = 0.05
        opt.net_args.init_ell = 0.1
        opt.net_args.max_ell = 0.2
        opt.net_args.is_se3 = True
        opt.net_args.is_centerize = True
        opt.train_args.num_training_optimization_iters = 500

        #db_train = ModelNet40Alignment(opt)


    return opt
