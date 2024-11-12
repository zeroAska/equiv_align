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
from data_loader.tum import TumFormatDataLoader

from torch.utils.data import DataLoader
import torch
import pickle
import os
import os.path as osp
from .modelnet40 import ModelNet40, ModelNet40Alignment, ModelNet40VoxelSmooth

from .stream import ImageStream
from .stream import StereoStream
from .stream import RGBDStream
import ipdb

def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = {
                    'ModelNet40': (ModelNet40,),
                    'ModelNet40Alignment': (ModelNet40Alignment,)
                   }
    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)

def create_datastream_auto_split(opt):
    assert opt.exp_args.is_auto_split_dataset
    #if opt.exp_args.dataset_name == "ModelNet40Alignment":
    #    db_train = ModelNet40Alignment(opt)
    #elif opt.exp_args.dataset_name == 'eth3d':
    if opt.exp_args.dataset_name == 'eth3d':
        if opt.exp_args.run_mode != 'train' \
                and opt.exp_args.run_mode != 'val' \
                and opt.exp_args.run_mode != 'test' :
            db_test = TumFormatDataLoader(opt.exp_args.dataset_name,
                                          opt.exp_args.dataset_path,
                                          run_mode=opt.exp_args.run_mode,
                                          covis_thresh=opt.exp_args.odom_covis_thresh,
                                          num_point=opt.exp_args.num_point,
                                          rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                          use_gt_init=opt.exp_args.use_gt_init,
                                          is_inlier_only=opt.exp_args.is_inlier_only,
                                          edge_only=opt.exp_args.edge_only,
                                          max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)
            db_val = TumFormatDataLoader(opt.exp_args.dataset_name,
                                          opt.exp_args.dataset_path,
                                          run_mode=opt.exp_args.run_mode,
                                          covis_thresh=opt.exp_args.odom_covis_thresh,
                                          num_point=opt.exp_args.num_point,
                                          rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                          use_gt_init=opt.exp_args.use_gt_init,
                                          is_inlier_only=opt.exp_args.is_inlier_only,
                                          edge_only=opt.exp_args.edge_only,
                                         max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)                                         
            db_train = TumFormatDataLoader(opt.exp_args.dataset_name,
                                          opt.exp_args.dataset_path,
                                          run_mode=opt.exp_args.run_mode,
                                          covis_thresh=opt.exp_args.odom_covis_thresh,
                                          num_point=opt.exp_args.num_point,
                                          rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                          use_gt_init=opt.exp_args.use_gt_init,
                                          is_inlier_only=opt.exp_args.is_inlier_only,

                                           edge_only=opt.exp_args.edge_only,

                                           max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)
        
        
        
        else:
            db_train = TumFormatDataLoader(opt.exp_args.dataset_name,
                                           opt.exp_args.dataset_path,
                                           run_mode='train',
                                           covis_thresh=opt.exp_args.odom_covis_thresh,
                                           num_point=opt.exp_args.num_point,
                                           rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                           use_gt_init=opt.exp_args.use_gt_init,
                                           is_inlier_only=opt.exp_args.is_inlier_only,
                                           edge_only=opt.exp_args.edge_only,
                                                                                     max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)
            db_val = TumFormatDataLoader(opt.exp_args.dataset_name,
                                         opt.exp_args.dataset_path,
                                         run_mode='val',
                                         covis_thresh=opt.exp_args.odom_covis_thresh,
                                         num_point=opt.exp_args.num_point,
                                         rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                         use_gt_init=opt.exp_args.use_gt_init,
                                         is_inlier_only=opt.exp_args.is_inlier_only,
                                         edge_only=opt.exp_args.edge_only,
                                         max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)
            db_test = TumFormatDataLoader(opt.exp_args.dataset_name,
                                          opt.exp_args.dataset_path,
                                          run_mode='test',
                                          covis_thresh=opt.exp_args.odom_covis_thresh,
                                          num_point=opt.exp_args.num_point,
                                          rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                          use_gt_init=opt.exp_args.use_gt_init,
                                          is_inlier_only=opt.exp_args.is_inlier_only,
                                          edge_only=opt.exp_args.edge_only,
                                          max_pairs_per_seq=opt.exp_args.max_pairs_per_seq)

                                      
    else:
        assert False

    #target_batch_size = (opt.exp_args.batch_size if dataset.run_mode != 'test' else len(opt.exp_args.gpus.split(",")))
    #print("batch size is ", batch_size)
    create_stream = lambda dataset: DataLoader(dataset,
                                               shuffle=(opt.exp_args.run_mode != 'eval_traj'),
                                               batch_size=1,
                                               num_workers=opt.exp_args.num_workers)


    if opt.exp_args.is_overfitting:
        #train_stream, val_stream = map(create_stream, [val_dataset, val_dataset])
        #val_stream = map(create_stream, [db_val])

        val_stream = DataLoader(dataset,
                                shuffle=True,
                                batch_size=opt.exp_args.batch_size,
                                num_workers=opt.exp_args.num_workers)
        test_stream = val_stream
        train_stream = val_stream
        
    else:
        train_stream, val_stream, test_stream = map(create_stream, [db_train, db_val, db_test])
    return train_stream, val_stream, test_stream
    

def create_datastream_manual_split(opt):

    assert not opt.exp_args.is_auto_split_dataset

    if opt.exp_args.dataset_name == "ModelNet40":
        db = ModelNet40(opt)
    elif opt.exp_args.dataset_name == "ModelNet40Alignment":
        db = ModelNet40Alignment(opt)
    elif opt.exp_args.dataset_name == "ModelNet40VoxelSmooth":
        db = ModelNet40VoxelSmooth(opt)
    elif opt.exp_args.dataset_name == 'eth3d':
        db = TumFormatDataLoader(opt.exp_args.dataset_name,
                                 opt.exp_args.dataset_path,
                                 run_mode='train',
                                 covis_thresh=opt.exp_args.odom_covis_thresh,
                                 num_point=opt.exp_args.num_point,
                                 rand_rotation_degree=opt.exp_args.max_rotation_degree,
                                 use_gt_init=opt.exp_args.use_gt_init,
                                 is_inlier_only=opt.exp_args.is_inlier_only,
                                 edge_only=opt.exp_args.edge_only)
    else:
        assert False

    create_stream = lambda dataset: DataLoader(dataset,
                                               shuffle=True,
                                               batch_size=opt.exp_args.batch_size,
                                               num_workers=opt.exp_args.num_workers) if len(dataset) > 0 else None


    train_size = int(opt.exp_args.train_frac * len(db))
    val_size = int(opt.exp_args.val_frac * len(db))
    test_size = len(db) - train_size - val_size
    
    if (opt.exp_args.is_overfitting):
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(db, [train_size, val_size, test_size])
        train_stream, val_stream, test_stream = map(create_stream, [val_dataset, val_dataset, val_dataset])
        
    else:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(db, [train_size, val_size, test_size])
        train_stream, val_stream, test_stream = map(create_stream, [train_dataset, val_dataset, test_dataset])

    return train_stream, val_stream, test_stream

def create_datastream(opt, **kwargs):
    if opt.exp_args.is_auto_split_dataset:
        return create_datastream_auto_split(opt)
    else:
        return create_datastream_manual_split(opt)


def create_pointcloud_stream(dataset_path, batch_size, num_workers, **kwargs):
    from torch.utils.data import DataLoader

    db = ImageStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    

def create_imagestream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = ImageStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_stereostream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = StereoStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_rgbdstream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = RGBDStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

