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
import open3d as o3d
import sys, os
import pypose as pp
import numpy as np
from baseline.baseline import baseline
from model.metrics.metrics import pose_log_norm, pose_Fro_norm
from data_loader.vgtk_utils import rotate_point_cloud, translate_point_cloud, crop_2d_array
from data_loader.gmm_noise import GmmSample
from model.utils import save_two_pc_transformed
import point_cloud_utils as pcu
import ipdb
import time

def batch_exp(baseline_name,
              pcd_file_name,
              max_degree,
              inlier_ratio,
              outlier_uniform_range,
              crop_ratio,
              num_runs,
              num_pts,
              suffix):

    pcd = o3d.io.read_point_cloud(pcd_file_name)
    pc = np.asarray(pcd.points)
    print("read pc shape ",pc.shape)

    log_folder = "result_global_"+str(max_degree)+'_'+str(inlier_ratio)+"_"+str(crop_ratio)+"_"+str(num_pts)+"_"+suffix
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_err_file = log_folder + "/log_err_" + baseline_name+ str(num_pts)+ ".txt"
    fro_err_file = log_folder + "/fro_err_" + baseline_name+ str(num_pts)+".txt"
    time_file = log_folder + "/time_"+baseline_name +"_"+str(num_pts)+ ".txt"
    
    with open(log_err_file, 'w') as f:
        pass
    with open(fro_err_file, 'w') as f:
        pass
    with open(time_file, 'w') as f:
        pass

    gmm_sample = GmmSample(inlier_ratio, 0.01,
                           outlier_uniform_range, pc.shape[0])
    

    print("Start running...")
    for r in range(num_runs):
        curr_folder = log_folder + "/" + str(r)
        if not os.path.exists(log_folder + "/"+str(r)):
            os.makedirs(log_folder + "/" + str(r))

        
        # create gt and transform points
        rotated_pc, R = rotate_point_cloud(pc, max_degree = max_degree)
        pc_target, t  = translate_point_cloud(rotated_pc, max_translation_norm=0.5)

        pc1 = gmm_sample.sample(pc, pcu.estimate_point_cloud_normals_knn(pc, 16)[1])
        pc2 = gmm_sample.sample(pc_target, pcu.estimate_point_cloud_normals_knn(pc_target, 16)[1])
        
        #pc1 = crop_2d_array(pc1, crop_ratio)
        #pc2 = crop_2d_array(pc2, crop_ratio)

        pc1_o3d = o3d.geometry.PointCloud()
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
        pc1 = np.asarray(pc1_o3d.farthest_point_down_sample(num_pts).points)
        pc2_o3d = o3d.geometry.PointCloud()
        pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
        pc2 = np.asarray(pc2_o3d.farthest_point_down_sample(num_pts).points)
        

        T_gt_inv = np.eye(4)
        T_gt_inv[:3,:3] = R
        T_gt_inv[:3, 3] = t
        T_gt = np.linalg.inv(T_gt_inv)        
        gt = pp.from_matrix(T_gt, pp.SE3_type)

        #np.save(curr_folder+  '/pc1.npy', pc1)
        #np.save(curr_folder+ '/pc2.npy', pc2)

        T_init = pp.identity_SE3()
        #save_two_pc_transformed(pc1, None, pc2, None, T_init.matrix(),
        #                        name_prefix=curr_folder+"/before_",
        #                        is_auto_assign_color=True)        
        
        time_start = time.perf_counter()
        
        result = baseline(baseline_name, pc1, pc2, T_init)[0]


        time_end = time.perf_counter()
        time_curr = time_end - time_start
        
        log_err = pose_log_norm(result, gt, pp.SE3_type)
        fro_err = pose_Fro_norm(result, gt, pp.SE3_type)
        
        with open(log_err_file, 'a') as f:
            f.write("{}\n".format(log_err[0].item()))
        with open(fro_err_file, 'a') as f:
            f.write("{}\n".format(fro_err[0].item()))
        with open(time_file, 'a') as f:
            f.write("{}\n".format(time_curr))
            
        #save_two_pc_transformed(pc1, None, pc2, None, result[0].matrix(),
        #                        name_prefix=curr_folder+"/after_",
        #                        is_auto_assign_color=True)        



#if __name__ == '__main__':

    

def single_exp(baseline_name, #pcd_file_name,
               max_degree,
               inlier_ratio,
               outlier_uniform_range,
               crop_ratio,
               num_runs,
               num_pts,
               suffix):
    
    log_folder = "result_global_"+str(max_degree)+'_'+str(inlier_ratio)+"_"+str(crop_ratio)+suffix
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_err_file = log_folder + "/log_err_" + baseline_name+"_"+str(num_pts)+".txt"
    fro_err_file = log_folder + "/fro_err_" + baseline_name+"_"+str(num_pts)+".txt"
    time_file = log_folder + "/time_"+baseline_name +"_"+str(num_pts)+ ".txt"
    with open(log_err_file, 'w') as f:
        pass
    with open(fro_err_file, 'w') as f:
        pass
    with open(time_file, 'w') as f:
        pass

    for r in range(num_runs):
        
        curr_folder = log_folder + "/" + str(r)
        if not os.path.exists(log_folder + "/"+str(r)):
            print("Error: {} folder doesn't exist".format(curr_folder))
        #    os.makedirs(log_folder + "/" + str(r))
        #pc1_o3d = o3d.io.read_point_cloud(curr_folder+"/0.pcd")
        #pc2_o3d = o3d.io.read_point_cloud(curr_folder+"/1.pcd")
        #ipdb.set_trace() 
        #pc1_o3d = pc1_o3d.farthest_point_down_sample(num_pts)
        #pc2_o3d = pc2_o3d.farthest_point_down_sample(num_pts)

        #pc1 = np.asarray(pc1_o3d.points)
        #pc2 = np.asarray(pc2_o3d.points)
        
        
        # create gt and transform points
        rotated_pc, R = rotate_point_cloud(pc, max_degree = max_degree)
        pc_target, t  = translate_point_cloud(rotated_pc, max_translation_norm=0.5)

        pc1 = gmm_sample.sample(pc, pcu.estimate_point_cloud_normals_knn(pc, 16)[1])
        pc2 = gmm_sample.sample(pc_target, pcu.estimate_point_cloud_normals_knn(pc_target, 16)[1])
        
        pc1 = crop_2d_array(pc1, crop_ratio)
        pc2 = crop_2d_array(pc2, crop_ratio)

        T_gt_inv = np.eye(4)
        T_gt_inv[:3,:3] = R
        T_gt_inv[:3, 3] = t
        T_gt = np.linalg.inv(T_gt_inv)        
        gt = pp.from_matrix(T_gt, pp.SE3_type)
        #gt = pp.from_matrix(np.reshape(np.genfromtxt(curr_folder + "/gt_poses.txt")[1,:], (4,4)), ltype=pp.SE3_type).Inv()
        #np.save(curr_folder+  '/pc1.npy', pc1)
        #np.save(curr_folder+ '/pc2.npy', pc2)

        T_init = pp.identity_SE3()
        #save_two_pc_transformed(pc1, None, pc2, None, T_init.matrix(),
        #                        name_prefix=curr_folder+"/before_"+baseline_name,
        #                        is_auto_assign_color=True)

        #save_two_pc_transformed(pc1, None, pc2, None, gt.matrix(),
        #                        name_prefix=curr_folder+"/gt_"+baseline_name,
        #                        is_auto_assign_color=True)        
        
        time_start = time.perf_counter()
        
        result = baseline(baseline_name, pc1, pc2, T_init)[0]

        time_end = time.perf_counter()
        time_curr = time_end - time_start

        log_err = pose_log_norm(result, gt, pp.SE3_type)
        fro_err = pose_Fro_norm(result, gt, pp.SE3_type)
        
        with open(log_err_file, 'a') as f:
            f.write("{}\n".format(log_err[0].item()))
        with open(fro_err_file, 'a') as f:
            f.write("{}\n".format(fro_err[0].item()))
        with open(time_file, 'a') as f:
            f.write("{}\n".format(time_curr))
        #save_two_pc_transformed(pc1, None, pc2, None, result[0].matrix(),
        #                        name_prefix=curr_folder+"/after_"+baseline_name,
        #                        is_auto_assign_color=True)        


if __name__ == "__main__":

    pcd_file_name = 'data_loader/bunny.pcd'
    outlier_uniform_range = 0.1
    suffix = 'apr19_time' # 'jan10' 
    num_runs = 10
    
    for baseline_name in ['fgr', 'fpfh']:
        for max_degree in [180, 90]:
            for outlier_ratio  in [0.0, 0.125]: #0.25, 0.375, 0.5]:
                for crop_ratio in [ 0.0, 0.125]: #, 0.25, 0.375, 0.5]:
                    for num_pts in [4000, 8000]: #[250, 500, 1000, 2000]:
                        #main(baseline_name,
                        inlier_ratio = 1 - outlier_ratio
                        batch_exp(baseline_name,#pcd_file_name,
                                  pcd_file_name,
                                   max_degree,
                                   inlier_ratio,
                                   outlier_uniform_range,
                                   crop_ratio,
                                   num_runs,
                                   num_pts,
                                   suffix)
    
    
