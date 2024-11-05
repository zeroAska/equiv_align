from tqdm import tqdm
from data_loader.base import RegistrationDataset
import getpass
import ipdb
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from data_loader.vgtk_utils import rotate_point_cloud
import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import glob
#from .augmentation import RGBDAugmentor
from data_loader.rgbd_utils import *
from model.option import gen_options
import ipdb
import open3d as o3d
import open3d
from open3d.geometry import VoxelGrid, PointCloud


class TumFormatDataLoader(RegistrationDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5000.0


    def __init__(self,
                 name,
                 dataset_path,
                 run_mode='train',
                 access_seq='random',
                 covis_thresh=0.95,
                 num_point=1024,
                 #is_eval_traj=False,
                 #eval_traj_seq=[],
                 seq_list=None,
                 rand_rotation_degree=None,
                 edge_only=False,
                 use_gt_init=False,
                 is_inlier_only=False):

        super(TumFormatDataLoader, self).__init__(name, dataset_path, run_mode)        

        self.n_frames = 2
        self.covisibility_inlier_thresh = covis_thresh
        self.num_point = num_point

        #self.is_eval_traj = is_eval_traj
        self.rand_rotation_degree = rand_rotation_degree
        self.use_gt_init = use_gt_init
        self.is_inlier_only = is_inlier_only

        self.access_seq=access_seq        
        self.edge_only = edge_only


        cur_path = osp.dirname(osp.abspath(__file__))        
        if not os.path.isdir(osp.join(cur_path, 'scene_cache')):
            os.mkdir(osp.join(cur_path, 'scene_cache'))
        if not os.path.isdir(osp.join(cur_path, 'dataindex_cache')):
            os.mkdir(osp.join(cur_path, 'dataindex_cache'))
        self.scene_cache_path = osp.join(cur_path, 'scene_cache', 'scene_cache.{}.pickle'.format(self.run_mode))
        self.dataindex_cache_path = osp.join(cur_path, 'dataindex_cache', 'dataindex_cache.{}.pickle'.format(self.run_mode))
        self._load_index(self.scene_cache_path, self.dataindex_cache_path)

        
        


    def set_is_use_gt_init(self, is_use_gt):
        self.use_gt_init = is_use_gt

    def save_index(self, path_identifier):
        ### get paths for the two index
        #cur_path = osp.dirname(osp.abspath(__file__))
        scene_cache_path = ( 'scene_cache.'+path_identifier+ '.pickle')
        dataindex_cache_path = 'dataindex_cache.'+path_identifier+'.pickle'

        ### load
        with open(scene_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(scene_cache_path))
            pickle.dump((self.scene_info,), cachefile)

        with open(dataindex_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(dataindex_cache_path))
            pickle.dump((self.dataset_index,), cachefile)

    def set_access_seq(self, new_seq):
        if new_seq == 'random':
            self.access_seq = new_seq
        elif new_seq in self.dataset_index:
            self.access_seq = new_seq

    def read_and_check_frame(self, id, images_list, depths_list, poses_list, intrinsics_list):
        color1 = self.__class__.image_read(images_list[id])
        depth1 = self.__class__.depth_read(depths_list[id])
        pose1 = xyzquat_to_pose_mat(poses_list[id])

        if isinstance(intrinsics_list, list):
            intrinsics = intrinsics_list[id]
            if intrinsics.size == 4:
                fx = intrinsics[ 0]
                fy = intrinsics[ 1]
                cx = intrinsics[ 2]
                cy = intrinsics[ 3]
                K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
            else:
                K = intrinsics
        else:
            intrinsics = intrinsics_list
            if intrinsics.size == 4:
                fx = intrinsics[ 0]
                fy = intrinsics[ 1]
                cx = intrinsics[ 2]
                cy = intrinsics[ 3]
                K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
            else:
                K = intrinsics

        xyz1, color_pc1 = self.rgbd_to_point_cloud( color1, depth1, K, target_num_pts=self.num_point, is_downsample=True,
                                                    edge_only=self.edge_only, color_path = images_list[id],
                                                    depth_path=depths_list[id])
        return xyz1, color_pc1, color1, depth1
        
        

    def check_frame_pair_valid(self, scene, i, j, images_list, depths_list, poses_list, intrinsics_list, frame_cache):

        if i in frame_cache:
            xyz1 = frame_cache[i]
        else:
            xyz1, _, color1, depth1 = self.read_and_check_frame(i, images_list, depths_list, poses_list, intrinsics_list)
            frame_cache[i] = xyz1

        if j in frame_cache:
            xyz2 = frame_cache[j]
        else:
            xyz2, _, color2, depth2 = self.read_and_check_frame(j, images_list, depths_list, poses_list, intrinsics_list)
            frame_cache[j] = xyz2

        """
        pose_i = xyzquat_to_pose_mat(poses_list[i])
        pose_j = xyzquat_to_pose_mat(poses_list[j])
        pose_fj_to_fi = np.linalg.inv(pose_j) @ pose_i               
        if (np.linalg.norm(pose_fj_to_fi[:3, 3]) > 1.0 or \
            np.linalg.norm(Rotation.from_matrix(pose_fj_to_fi[:3, :3]).as_euler('xyz', degrees=True)) > 90 ):
            continue
        inlier_rate = min(self.project_in_view_rate(xyz1,
                                                    pose_fj_to_fi,
                                                    intrinsics),
                          self.project_in_view_rate(xyz2,
                                                    np.linalg.inv(pose_fj_to_fi),
                                                    intrinsics))                                  
        if inlier_rate > inlier_thresh:
            inlier_pairs.append((i,j))
            if is_logging:
                img_i = self.__class__.image_read(images[i])
                img_j = self.__class__.image_read(images[j])
                img_ij = np.concatenate((img_i, img_j), axis=1)
                cv2.imwrite(osp.join(scene_name , str(i)+"_"+str(j)+".png"), img_ij)
        
        """
            
        return xyz1 is not None and \
            xyz1.shape[0] == self.num_point and \
            xyz2 is not None and \
            xyz2.shape[0] == self.num_point
            
    
    
    def filter_scene_cache(self, scene_cache_path):

        new_scene_info = {}
        for scene_path, scene in self.scene_info.items():

            images_list = scene['images']
            depths_list = scene['depths']
            poses_list =  scene['poses']
            intrinsics_list = scene['intrinsics']
            print("before filter size is ",len(self.scene_info[scene_path]['frame_pairs']))
            
            new_scene_info[scene_path] = scene

            frame_cache = {}
            
            new_scene_info[scene_path]['frame_pairs'] = [(pair[0], pair[1]) for pair in tqdm(scene['frame_pairs']) if self.check_frame_pair_valid(scene, pair[0],pair[1], images_list, depths_list, poses_list, intrinsics_list, frame_cache)]
            print("after filter size is ",len(new_scene_info[scene_path]['frame_pairs']))
        with open(scene_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(scene_cache_path))                
            pickle.dump((new_scene_info,), cachefile)
        self.scene_info = new_scene_info
            
    def get_seq_ground_truth(self, seq_name):
        for seq_full_name in self.scene_info.keys():
            if seq_full_name.endswith(seq_name):
                return len(self.scene_info[seq_full_name]['gt_poses_all'])
        return None
        
    def get_scene_names(self):
        return self.scene_info.keys()

    def get_seq_num_frames(self, seq_name):
        for seq_full_name in self.scene_info.keys():
            if seq_full_name.endswith(seq_name):
                return len(self.scene_info[seq_full_name]['images'])
        return 0

    def _build_dataset(self):
        print("Building TUM dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.seq_path_all,  '*'))

        if self.run_mode == 'eval_traj': #self.is_eval_traj:
            frame_rate = 1
        else:
            frame_rate = 1

        for scene in tqdm(sorted(scenes)):

            print("processing {}".format(scene))
            scene_path = scene
            images, depths, poses, intrinsics, tstamps, gt_poses_all = loadtum(scene_path, frame_rate)
            
            # graph of co-visible frames based on flow
            fx = intrinsics[0][ 0]
            fy = intrinsics[0][ 1]
            cx = intrinsics[0][ 2]
            cy = intrinsics[0][ 3]
            K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
            scene_name = scene.split('/')[-1]
            #if (scene_name != "cables_3" and scene_name != 'plant_1'):
            #    continue
            #ipdb.set_trace()
            frame_pairs = self.build_frame_pair(scene_name, poses,images, depths, intrinsics[0], self.covisibility_inlier_thresh, False)
            print("# of frame_pairs in {} is {}".format(scene, len(frame_pairs)))

            scene = '/'.join(scene.split('/'))

            scene_info[scene] = {'images': images, 'depths': depths, 
                                 'poses': poses, 'intrinsics': intrinsics, 'frame_pairs': frame_pairs,
                                 'gt_poses_all': gt_poses_all}

            
        return scene_info

    @staticmethod
    def rgbd_to_point_cloud_uniform_downsample(color_path, depth_path, intrinsics, target_num_pts=None):
        if intrinsics.size != 4:
            intrinsics = np.array([intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]])
            
        pcd = open3d_pc_from_rgbd(color_path, depth_path, intrinsics,
                                  target_num_pts)
        return pcd

    @staticmethod
    def rgbd_to_point_cloud_edge_downsample(color, depth, intrinsics, target_num_pts=None):

        if intrinsics.size == 4:
            fx = intrinsics[ 0]
            fy = intrinsics[ 1]
            cx = intrinsics[ 2]
            cy = intrinsics[ 3]
            intrinsics = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])


        
        
        h, w = depth.shape
        coord = np.mgrid[0:h, 0:w].astype(float) + 0.5
        ones = np.ones((h, w, 1)).astype(float)
        xyz = np.concatenate((np.expand_dims(coord[0], axis=-1),
                              np.expand_dims(coord[1], axis=-1),
                              ones), axis=-1)
        #xyz = np.transpose(xyz, (1,0,2)).reshape((-1, 3))
        xyz = xyz.reshape((-1, 3))

        depth_2d = np.expand_dims(depth, axis=-1)
        depth_2d = np.concatenate((depth_2d, depth_2d, depth_2d ), axis=-1)
        #depth_2d = np.transpose(depth_2d, (1,0,2)).reshape((-1, 3))
        depth_2d = depth_2d.reshape((-1, 3))
        xyz = xyz * depth_2d

        valid_rows = np.logical_and(depth_2d>0,  ~np.isnan(xyz), np.isfinite(xyz) ).all(axis=1)
        valid_cols = np.array([True, True, True])
        if color is not None: #and is_downsample is True:
            edges = cv2.Canny(color, 50,100)
            edges = edges > 0
            edges = np.transpose(edges,(1,0)).reshape((-1,))
            valid_rows = np.logical_and(valid_rows, edges)
        
        xyz = xyz[valid_rows][:, valid_cols]
        xyz = (np.linalg.inv(intrinsics) @ xyz.transpose()).transpose()
        
        if color is not None:
            # color_pc = np.transpose(color, (1,0,2)).reshape((-1, 3))            
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            color_pc = color.reshape((-1, 3))           
            color_pc = color_pc[valid_rows][:,valid_cols].astype(float) / 255.0
        else:
            color_pc = None

        if target_num_pts is not None:
            colored_pt_cloud = PointCloud()
            colored_pt_cloud.points = o3d.utility.Vector3dVector(xyz)
            if color is not None:
                colored_pt_cloud.colors = o3d.utility.Vector3dVector(color_pc)
            if xyz.shape[0] < target_num_pts:
                print("Warning: xyz shape {} less than 1024".format(xyz.shape))
                return None, None
            colored_pt_cloud = colored_pt_cloud.farthest_point_down_sample(target_num_pts)
            xyz = np.asarray(colored_pt_cloud.points)
            color_pc = np.asarray(colored_pt_cloud.colors)
        
        assert(xyz is not None) 
            
        return xyz, color_pc


        
    @staticmethod
    def rgbd_to_point_cloud(color, depth, intrinsics, target_num_pts=None, is_downsample=False,
                            edge_only=True, color_path=None, depth_path=None):
        if edge_only:
            return TumFormatDataLoader.rgbd_to_point_cloud_edge_downsample(color, depth, intrinsics, target_num_pts=target_num_pts)
        else:

            pc_open3d = TumFormatDataLoader.rgbd_to_point_cloud_uniform_downsample(color_path, depth_path,
                                                                                   intrinsics, target_num_pts=target_num_pts)
            return np.asarray(pc_open3d.points), np.asarray(pc_open3d.colors)
            
        
    @staticmethod
    def downsample_farthest_colored_pc(xyz1, color_pc1, target_num_pts):
        colored_pt_cloud1 = PointCloud()
        colored_pt_cloud1.points = o3d.utility.Vector3dVector(xyz1)
        colored_pt_cloud1.colors = o3d.utility.Vector3dVector(color_pc1)
        if xyz1.shape[0] < target_num_pts:
            print("Warning: xyz shape {} less than 1024".format(xyz1.shape))
            #return None, None
        colored_pt_cloud1 = colored_pt_cloud1.farthest_point_down_sample(target_num_pts)
        xyz1 = np.asarray(colored_pt_cloud1.points)
        color_pc1 = np.asarray(colored_pt_cloud1.colors)
        return xyz1, color_pc1
        
        

    def project_in_view_rate(self, xyz1,  pose_f2_to_f1, intrinsics,
                             h, w):
        if xyz1 is None or xyz1.shape[0] < self.num_point:
            return 0

        if intrinsics.size == 4:
            fx = intrinsics[ 0]
            fy = intrinsics[ 1]
            cx = intrinsics[ 2]
            cy = intrinsics[ 3]
            intrinsics = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
            
        #ipdb.set_trace()
        #depth2_flat = 1.0 / depth2.reshape((-1,))
        xyz2 = (intrinsics @ (pose_f2_to_f1[:3,:3]@  xyz1.transpose()) + \
                              np.broadcast_to(np.expand_dims(  pose_f2_to_f1[:3, 3], axis=-1) , (3,xyz1.shape[0]))).transpose()

        uv2 = np.zeros_like(xyz2)
        uv2[:, 0] = xyz2[:, 0] / xyz2[:, 2]
        uv2[:, 1] = xyz2[:, 1] / xyz2[:, 2]
        uv2[:, 2] = 1.0 #xyz2[:, 2] / xyz2[:, 2]

        inlier = np.logical_and.reduce((uv2[:,0] >= 0, uv2[:,0] < w, \
                                        uv2[:,1] >= 0 , uv2[:,1] < h,\
                                        xyz2[:,2] > 0))

        #print("inlier ratio : {}".format(float(inlier)/xyz1.shape[0])) 
        return float(inlier.sum()) / xyz1.shape[0], inlier#float(self.w * self.h)
        

    def build_frame_pair(self,scene_name, poses, images, depths, intrinsics, inlier_thresh, is_logging=False):
        """ 
        compute shared view between all pairs of frames, by projecting the 
        depth between each other and calculate how much overlap
        """
        if is_logging:
            os.mkdir(scene_name)

        num_poses = len(poses)
        #if self.run_mode == 'train' or self.run_mode == 'val':
        if self.run_mode == 'eval_traj': #self.is_eval_traj:
            candidate_pairs = [(i, i+1) for i in range(num_poses-1)]
        else:
            candidate_pairs = [(i, j) for i in range(num_poses-1) for j in range(i+1, min(num_poses, i+10), 1)]

        frame_cache = {}
        poses = np.array(poses)
        intrinsics = np.array(intrinsics) #/ f
        fx = intrinsics[ 0]
        fy = intrinsics[ 1]
        cx = intrinsics[ 2]
        cy = intrinsics[ 3]
        K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])

        inlier_pairs = []
        #ipdb.set_trace()
        for i,j in tqdm(candidate_pairs):


            if i in frame_cache:
                xyz1 = frame_cache[i]
            else:
                xyz1, color_pc1, color1, depth1 = self.read_and_check_frame(i, images, depths, poses, intrinsics)
                frame_cache[i] = xyz1

            if j in frame_cache:
                xyz2 = frame_cache[j]
            else:
                if j >= len(images):
                    ipdb.set_trace()
                xyz2, color_pc2, color2, depth2 = self.read_and_check_frame(j, images, depths, poses, intrinsics)
                frame_cache[j] = xyz2
            if xyz1 is None or xyz2 is None or np.isnan(xyz1).any() or np.isnan(xyz2).any()\
                    or xyz1.shape[0] < self.num_point or xyz2.shape[0] < self.num_point:
                continue
            
            h, w = depth1.shape
            pose_i = xyzquat_to_pose_mat(poses[i])
            pose_j = xyzquat_to_pose_mat(poses[j])
            pose_fj_to_fi = np.linalg.inv(pose_j) @ pose_i               
            if (np.linalg.norm(pose_fj_to_fi[:3, 3]) > 1.0 or \
                np.linalg.norm(Rotation.from_matrix(pose_fj_to_fi[:3, :3]).as_euler('xyz', degrees=True)) > 90 ):
                print("the pair {}-{} is removed because angle is too large or translation too large")
                continue
            inlier_rate = min(self.project_in_view_rate(xyz1,
                                                        pose_fj_to_fi,
                                                        intrinsics, h, w)[0],
                              self.project_in_view_rate(xyz2,
                                                        np.linalg.inv(pose_fj_to_fi),
                                                        intrinsics, h, w)[0])                                  
            if inlier_rate > inlier_thresh or j-i == 1:
                inlier_pairs.append((i,j))
                if is_logging:
                    img_i = self.__class__.image_read(images[i])
                    img_j = self.__class__.image_read(images[j])
                    img_ij = np.concatenate((img_i, img_j), axis=1)
                    cv2.imwrite(osp.join(scene_name , str(i)+"_"+str(j)+".png"), img_ij)
                print("{}: inlier pair between {} and {} is {}. addeded".format(scene_name, i,j, inlier_rate))
            #else:
            #ipdb.set_trace()

            #print("{}: inlier pair between {} and {} is too small {}. removed".format(scene_name, i,j, inlier_rate))
        
        return inlier_pairs
    
    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        if depth_file.endswith(".npy"):
            depth = np.load(depth_file)
        elif depth_file.endswith(".png"):
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED ).astype(float)
        depth = depth / TumFormatDataLoader.DEPTH_SCALE
        return depth

    @staticmethod
    def stack_two_pc(xyz1, color_pc1, xyz2, color_pc2,  pose_f1_to_f2, write_to_disk=True, name_prefix="", pose1=None, pose2=None):
        if xyz1 is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz1)
            pcd.colors = o3d.utility.Vector3dVector(color_pc1)
            if (write_to_disk):
                
                o3d.io.write_point_cloud(name_prefix+"pc1.ply", pcd)
                print("write to pc1.ply")

        if xyz2 is not None:                
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(xyz2)
            pcd2.colors = o3d.utility.Vector3dVector(color_pc2)
            if (write_to_disk):
                o3d.io.write_point_cloud(name_prefix+"pc2.ply", pcd2)
                print("write to pc2.ply")


        if xyz1 is not None and xyz2 is not None:                
            xyz = np.concatenate((xyz1, xyz2), axis=0)
            color = np.concatenate((color_pc1, color_pc2), axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(color)
        
            if (write_to_disk):
                o3d.io.write_point_cloud(name_prefix+"pc_init_12.ply", pcd)
                print("write to pc_init_12.ply")
                

        if xyz1 is not None and xyz2 is not None and pose_f1_to_f2 is not None:
            xyz2 = (pose_f1_to_f2[:3,:3] @ xyz2.transpose() +\
                    np.broadcast_to(np.expand_dims(  pose_f1_to_f2[:3, 3], axis=-1) , (3,xyz2.shape[0]))).transpose()
            xyz_full = np.concatenate((xyz1, xyz2), axis=0)
            color = np.concatenate((color_pc1, color_pc2), axis=0)
            pcd_trans = o3d.geometry.PointCloud()
            pcd_trans.points = o3d.utility.Vector3dVector(xyz_full)
            pcd_trans.colors = o3d.utility.Vector3dVector(color)
        
            if (write_to_disk):
                o3d.io.write_point_cloud(name_prefix+"pc_12.ply", pcd_trans)
                print("write to pc_12.ply")

    def get_data_pair(self, scene_id, images_list, depths_list, poses_list, intrinsics_list, id1, id2):
        
        color1 = self.__class__.image_read(images_list[id1])
        color2 = self.__class__.image_read(images_list[id2])
        depth1 = self.__class__.depth_read(depths_list[id1])
        depth2 = self.__class__.depth_read(depths_list[id2])
        pose1 = xyzquat_to_pose_mat(poses_list[id1])
        pose2 = xyzquat_to_pose_mat(poses_list[id2])
        pose_f1_to_f2 = np.linalg.inv(pose1) @ pose2
        intrinsics = intrinsics_list[id1]
        fx = intrinsics[ 0]
        fy = intrinsics[ 1]
        cx = intrinsics[ 2]
        cy = intrinsics[ 3]
        K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])
        h, w = depth1.shape


        if self.is_inlier_only == False:
            assert self.edge_only == False
            xyz1, color_pc1 = self.__class__.rgbd_to_point_cloud( color1, depth1, K if self.edge_only else intrinsics,
                                                                  target_num_pts=self.num_point
                                                                  , is_downsample=True, edge_only=self.edge_only,
                                                                  color_path= images_list[id1], depth_path=depths_list[id1])
            xyz2, color_pc2 = self.__class__.rgbd_to_point_cloud( color2, depth2, K if self.edge_only else intrinsics,
                                                                  target_num_pts=self.num_point
                                                                  , is_downsample=True, edge_only=self.edge_only, 
                                                                  color_path= images_list[id2], depth_path=depths_list[id2])                                                                 

            if self.rand_rotation_degree is not None and abs(self.rand_rotation_degree)>1e-4:
                xyz2, R_f2p_to_f2 = rotate_point_cloud(xyz2, max_degree = self.rand_rotation_degree)
                pose_f2p_to_f2 = np.eye(4)
                pose_f2p_to_f2[:3,:3] = R_f2p_to_f2
                pose_f1_to_f2p = pose_f1_to_f2 @ np.linalg.inv(pose_f2p_to_f2)
                pose_f1_to_f2 = pose_f1_to_f2p
        
            if self.use_gt_init:
                xyz2_new = (pose_f1_to_f2[:3,:3] @ xyz2.transpose() +\
                        np.broadcast_to(np.expand_dims(  pose_f1_to_f2[:3, 3], axis=-1) , (3,xyz2.shape[0]))).transpose()
                #print(pose_f1_to_f2)
                pose_f1_to_f2 = np.eye(4)
                #if True:
                #    save_color_ply(np.concatenate((xyz1, xyz2_new), axis=0), np.concatenate((color_pc1, color_pc2), axis=0), "_transform_stacked.ply")
                #    save_color_ply(np.concatenate((xyz1, xyz2), axis=0), np.concatenate((color_pc1, color_pc2), axis=0), "stacked.ply")
                #    save_color_ply(xyz1, color_pc1, "pc1.ply")
                #    save_color_ply(xyz2, color_pc2, "pc2.ply")
                xyz2 = xyz2_new

        else:
            xyz1, color_pc1 = self.__class__.rgbd_to_point_cloud( color1, depth1, K if self.edge_only else intrinsics,
                                                                  is_downsample=False, edge_only=self.edge_only,
                                                                  color_path= images_list[id1], depth_path=depths_list[id1])
            xyz2, color_pc2 = self.__class__.rgbd_to_point_cloud( color2, depth2, K if self.edge_only else intrinsics,
                                                                  is_downsample=False, edge_only=self.edge_only, 
                                                                  color_path= images_list[id2], depth_path=depths_list[id2])                                                                 
                                                    

            if self.rand_rotation_degree is not None and abs(self.rand_rotation_degree)>1e-4:
                xyz2, R_f2p_to_f2 = rotate_point_cloud(xyz2, max_degree = self.rand_rotation_degree)
                pose_f2p_to_f2 = np.eye(4)
                pose_f2p_to_f2[:3,:3] = R_f2p_to_f2
                pose_f1_to_f2p = pose_f1_to_f2 @ np.linalg.inv(pose_f2p_to_f2)
                pose_f1_to_f2 = pose_f1_to_f2p
        
            if self.use_gt_init:
                xyz2 = (pose_f1_to_f2[:3,:3] @ xyz2.transpose() +\
                        np.broadcast_to(np.expand_dims(  pose_f1_to_f2[:3, 3], axis=-1) , (3,xyz2.shape[0]))).transpose()
                pose_f1_to_f2 = np.eye(4)

            inlier1, inlier_inds1 = self.project_in_view_rate(xyz1,np.linalg.inv(pose_f1_to_f2), K, h, w)
            inlier2, inlier_inds2 = self.project_in_view_rate(xyz2,pose_f1_to_f2, K, h, w)
            xyz1 = xyz1[inlier_inds1, :]
            color_pc1 = color_pc1[inlier_inds1, :]                
            xyz2 = xyz2[inlier_inds2, :]
            color_pc2 = color_pc2[inlier_inds2, :]
            
            xyz1, color_pc1 = TumFormatDataLoader.downsample_farthest_colored_pc(xyz1, color_pc1, self.num_point)
            xyz2, color_pc2 = TumFormatDataLoader.downsample_farthest_colored_pc(xyz2, color_pc2, self.num_point)                
            

        assert xyz1 is not None
        assert xyz2 is not None
        assert color_pc1 is not None
        assert color_pc2 is not None
        assert pose_f1_to_f2 is not None
        assert scene_id is not None
        ### convert to torch
        shuffle1 = torch.randperm(xyz1.shape[0])
        shuffle2 = torch.randperm(xyz2.shape[0])
        in_dict = {
            'pc1': torch.from_numpy(xyz1.astype(np.float32))[shuffle1],
            'pc2': torch.from_numpy(xyz2.astype(np.float32))[shuffle2],            
            'color1': torch.from_numpy(color_pc1.astype(np.float32))[shuffle1],
            'color2': torch.from_numpy(color_pc2.astype(np.float32))[shuffle2],
            'T': torch.from_numpy(pose_f1_to_f2.astype(np.float32)),
            't': torch.from_numpy(pose_f1_to_f2[:3, 3].astype(np.float32)),
            'R': torch.from_numpy(pose_f1_to_f2[:3, :3].astype(np.float32)),
            'seq': scene_id,
            'pc1_id': id1,
            'pc2_id': id2,
            'intrinsic': K,
            'h': h,
            'w': w
        }

        #if True:
        if False:
        #if index == 0 :
            print("store staced point cloud id {} and {}".format(images_list[id1], images_list[id2]))
            print("relative pose from pc1 to pc2 is {}".format(pose_f1_to_f2))
            self.__class__.stack_two_pc(xyz1, color_pc1, xyz2, color_pc2, pose_f1_to_f2)
            cv2.imwrite("pc1.png", color1)
            cv2.imwrite("pc2.png", color2)

            #xyz1, color_pc1 = self.__class__.rgbd_to_point_cloud( color1, depth1, K,  is_downsample=False)
            #xyz2, color_pc2 = self.__class__.rgbd_to_point_cloud( color2, depth2, K,  is_downsample=False)
            #self.__class__.stack_two_pc(xyz1, color_pc1, xyz2, color_pc2, pose_f1_to_f2, name_prefix="full_", pose1=pose1, pose2=pose2)
            
        #if self.aug is not None:
        #    images, poses, disps, intrinsics = \
        #        self.aug(images, poses, disps, intrinsics)

        # scale scene
        #if len(disps[disps>0.01]) > 0:
        #    s = disps[disps>0.01].mean()
        #    disps = disps / s
        #    poses[...,:3] *= s

        return in_dict
        

    def downsample_dataindex(self, max_data_pair_num=1000):
        cur_path = osp.dirname(osp.abspath(__file__))        
        dataindex_cache_path = osp.join(cur_path, 'dataindex_cache', '{}.pickle'.format(self.run_mode))
        
        self.dataset_index = self._build_dataset_index( self.dataindex_cache_path, max_data_pair_num)

    def get_dataindex(self, dataset_index, index):
        scene_id, frame_pair = dataset_index[index]

        #if (self.access_seq != 'random' and not scene_id.endswith( self.access_seq) ):
        #    return None
        
        id1, id2 = frame_pair

        #frame_pairs = self.scene_info[scene_id]['frame_pairs']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        
        return self.get_data_pair(scene_id, images_list, depths_list, poses_list, intrinsics_list, id1, id2)
        #print("reading {} and {}".format(images_list[id1], images_list[id2]))
        

    def get_seq_index(self, scene_id, seq_index, index):


        #frame_pairs = self.scene_info[scene_id]['frame_pairs']
        images_list = seq_index['images']
        depths_list = seq_index['depths']
        poses_list = seq_index['poses']
        intrinsics_list = seq_index['intrinsics']
        frame_pair_list = seq_index['frame_pairs']
        
        id1, id2 = frame_pair_list[index]
        #print("reading {} and {}".format(images_list[id1], images_list[id2]))
        return get_data_pair(scene_id, images_list, depths_list, poses_list, intrinsics_list, id1, id2)        


    def __getitem__(self, index):
        if self.access_seq == 'random':
            #index_all_seq = {key: len(self.dataset_index[key]) for key in self.dataset_index.keys() }
            #curr_ind_remained = self.__len__()
            for key in self.dataset_index.keys():
                new_len = len(self.dataset_index[key])
                if index - new_len > 0:
                    index -= new_len
                else:
                    selected_key = key
            return self.get_dataindex(self.dataset_index[selected_key], index)
        else:
            index = index % len(self.dataset_index[self.access_seq])
            return self.get_dataindex(self.dataset_index[self.access_seq], index)


    def filter_nan_and_seq(self, target_seqs, is_filter_nan=False,
                           mode=''):
        non_nan_index = {}
        for seq in self.dataset_index.keys():
            if seq in target_seqs:
                non_nan_index[seq] = []
                for i in tqdm(range(len(self.dataset_index[seq]))):
                    d = self.get_dataindex(self.dataset_index[seq], i)
                    if is_filter_nan:                        
                        if torch.isnan(d['pc1']) or \
                           torch.isnan(d['pc2']) or \
                           torch.isnan(d['color1']) or \
                           torch.isnan(d['color2']) or \
                           torch.isnan(d['T']) or \
                           torch.isnan(d['t']) or \
                           torch.isnan(d['R']):
                            continue
                    non_nan_index[seq].append(d)
                   
        print("new dataindex len is ", len(non_nan_index))
        self.dataset_index = non_nan_index
        self.save_index(mode)        


            
        

                
                


def filter_index(mode='train', target_mode='train'):
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    opt.exp_args.odom_covis_thresh = 0.5
    opt.exp_args.edge_only = False
    
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, run_mode=mode,
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   num_point=opt.exp_args.num_point
                                   )

    if target_mode == 'train':
        target_seqs = {
            'cables_3',             
            'ceiling_1',
            'repetitive',
            #'planar_2',
            'einstein_2',
            'sfm_house_loop',
            'desk_3'    
        }
    elif target_mode == 'val':
        target_seqs = {
            'mannequin_3',
            'sfm_garden'
        }
    elif target_mode == 'test':
        target_seqs = {
            'sfm_lab_room_1',
            'plant_1',
            'sfm_bench',
            'table_3'
        }
    elif target_mode == 'overfit':
        target_seqs = {'sfm_lab_room_1'}
    elif target_mode == 'eval_traj':
        target_seqs = {
            #'cables_3',             
            #'ceiling_1',
            #'repetitive',
            #'einstein_2',
            #'sfm_house_loop',
            #'desk_3',    
            #'mannequin_3',
            #'sfm_garden',
            'sfm_lab_room_1',
            'plant_1',
            'sfm_bench',
            'table_3'
        }
    else:
        target_seqs = {target_mode}



    ## dataset length:
    # eth3d/training/cables_3 1336              
    # eth3d/training/ceiling_1 6780
    # eth3d/training/desk_3 5684
    # eth3d/training/einstein_2 8058
    # eth3d/training/mannequin_3 1474
    # eth3d/training/planar_2 4157

    # eth3d/training/repetitive 9759
    # eth3d/training/sfm_bench 1924
    # eth3d/training/sfm_garden 1441
    # eth3d/training/sfm_house_loop 919
    # eth3d/training/sfm_lab_room_1 540
    

    ### Train
    # eth3d/training/cables_3 1336              
    # eth3d/training/ceiling_1 6780
    # eth3d/training/repetitive 9759
    # eth3d/training/planar_2 4157
    # eth3d/training/einstein_2 8058
    # eth3d/training/sfm_house_loop 919

    ### val
    # eth3d/training/mannequin_3 1474
    # eth3d/training/desk_3 5684
    # eth3d/training/sfm_garden 1441

    ### Test
    # eth3d/training/sfm_lab_room_1 540
    # eth3d/training/plant_1 162 REMOVE
    # eth3d/training/sfm_bench 1924
    # table_3

    db_train.filter_nan_and_seq(target_seqs, mode=target_mode)

def filter_high_overlap_pairs(overlap_ratio, log_file):

    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+ getpass.getuser() + "/data/eth3d/"
    opt.exp_args.batch_size = 1
    opt.exp_args.odom_covis_thresh = overlap_ratio
    db = TumFormatDataLoader('eth3d',
                             opt.exp_args.dataset_path, 'train',
                             opt.exp_args.odom_covis_thresh,
                             opt.exp_args.num_point)
    #if is_filter:
    #    db.filter_scene_cache('data_loader/scene_cache/train_filtered.pickle')
    #    db._build_dataset_index('data_loader/dataindex_cache/train_filtered.pickle')
    
    train_stream = torch.utils.data.DataLoader(db, shuffle=False, batch_size=opt.exp_args.batch_size, num_workers=opt.exp_args.num_workers)
    high_overlaps = 0
    total = 0
    with open (log_file, 'w') as f:
        for p in (iter(train_stream)):
            pc1 = p['pc1'].squeeze().numpy()
            pc2 = p['pc2'].squeeze().numpy()
            color1 = p['color1'].squeeze().numpy()
            color2 = p['color2'].squeeze().numpy()
            T = p['T'].squeeze().numpy()
            K = p['intrinsic'].squeeze().numpy()

            #import ipdb; ipdb.set_trace()
            cx = K[0,2]
            cy = K[1,2]
            h = int(cy * 2)
            w = int(cx * 2)
            
            inlier_rate = min(db.project_in_view_rate(pc1,
                                                      np.linalg.inv(T),
                                                      K,
                                                      h, w)[0],
                              db.project_in_view_rate(pc2,
                                                      T,
                                                      K,
                                                      h, w)[0])
            f.write("{}\n".format(inlier_rate))
            if (inlier_rate > overlap_ratio):
                high_overlaps += 1
            elif total - high_overlaps == 1:
                print("Overlap: {}".format(inlier_rate))
                TumFormatDataLoader.stack_two_pc(pc1, color1, pc2, color2, T)
            total += 1

    print("{} out of {} pairs have higher than {} overlap".format(high_overlaps, total, overlap_ratio))
                              

        
        

def filter_out_invalid_pairs(is_filter):
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    db = TumFormatDataLoader('eth3d',
                             opt.exp_args.dataset_path, 'train',
                             opt.exp_args.odom_covis_thresh,
                             opt.exp_args.num_point)
    if is_filter:
        db.filter_scene_cache('data_loader/scene_cache/train_filtered.pickle')
        db._build_dataset_index('data_loader/dataindex_cache/train_filtered.pickle')
    
    train_stream = torch.utils.data.DataLoader(db, shuffle=False, batch_size=opt.exp_args.batch_size, num_workers=opt.exp_args.num_workers)
    print(next(iter(train_stream)))

def print_dataindex_overlap_all():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, 'train',
                                   opt.exp_args.odom_covis_thresh,
                                   opt.exp_args.num_point)
    print("train_len:", len(db_train.dataset_index))

def construct_traj_index():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.exp_args.odom_covis_thresh = 0.1
    opt.exp_args.edge_only = False
    #opt.exp_args.is_user_specified_trainvaltest = True
    #opt.exp_args.is_eval_traj = True
    
    #opt.batch_size = 1
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, run_mode='traj',
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   #is_eval_traj = opt.exp_args.is_eval_traj,
                                   num_point=opt.exp_args.num_point)

def construct_db_index():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.exp_args.odom_covis_thresh = 0.95
    opt.exp_args.edge_only = False
    #opt.batch_size = 1
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, run_mode='train',
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   num_point=opt.exp_args.num_point)


def balance_dataindex_seq_num():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    opt.exp_args.odom_covis_thresh = 0.95
    opt.exp_args.edge_only = False
    #opt.exp_args.is_user_specified_trainvaltest = True
    
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, 'train',
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   num_point=opt.exp_args.num_point)

    db_train.downsample_dataindex(1000)

    total = 0
    for key in db_train.scene_info:
        print(key, len(db_train.scene_info[key]['frame_pairs']))
        total += len(db_train.scene_info[key]['frame_pairs'])
    print("total pairs: ",total)
    
    
def print_dataindex_len():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    opt.exp_args.odom_covis_thresh = 0.95
    opt.exp_args.edge_only = False
    
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, 'train',
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   num_point=opt.exp_args.num_point)


    total = 0
    for key in db_train.scene_info:
        print(key, len(db_train.scene_info[key]['frame_pairs']))
        total += len(db_train.scene_info[key]['frame_pairs'])
    print("total pairs: ",total)

    #next(iter(db_train))
    #next(iter(db_train))
    #next(iter(db_train))
    #next(iter(db_train))
    
    #db_val = TumFormatDataLoader('eth3d',
    #                             opt.exp_args.dataset_path, 'val',
    #                             opt.exp_args.odom_covis_thresh,
    #                             opt.exp_args.num_point)
    #print("val_len:", len(db_val.dataset_index))    
    
    #db_test = TumFormatDataLoader('eth3d',
    #                              opt.exp_args.dataset_path, 'test',
    #                              opt.exp_args.odom_covis_thresh,
    #                              opt.exp_args.num_point)
    #print("test_len:", len(db_test.dataset_index))


def open3d_pc_from_rgbd(color_path1, depth_path1, intrinsics,
                        target_num_pts=None):
    color1 = o3d.io.read_image(color_path1)
    depth1 = o3d.io.read_image(depth_path1)
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(color1, depth1,
                                                          convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1,
        o3d.camera.PinholeCameraIntrinsic(np.array(color1).shape[1], np.array(color1).shape[0],
                                          intrinsics[0],intrinsics[1],intrinsics[2],intrinsics[3]))

    if target_num_pts is not None:
        pcd = pcd.farthest_point_down_sample(target_num_pts) if np.asarray(pcd.points).shape[0] >= target_num_pts else pcd

    return pcd


def open3d_stack_pc_from_rgbd(color_path1, depth_path1, color_path2, depth_path2,
                              pose_quat1, pose_quat2,  intrinsics):
    print("Read TUM dataset")
    pcd1 = open3d_pc_from_rgbd(color_path1, depth_path1, intrinsics, 1024)
    pcd2 = open3d_pc_from_rgbd(color_path2, depth_path2, intrinsics, 1024)

    pose1 = xyzquat_to_pose_mat(pose_quat1)
    pose2 = xyzquat_to_pose_mat(pose_quat2)
    pose_f1_to_f2 = (np.linalg.inv(pose1) @ pose2)

    pcd2 = pcd2.transform(pose_f1_to_f2)

    o3d.io.write_point_cloud("pc_1.ply", pcd1)
    o3d.io.write_point_cloud("pc_2.ply", pcd2)
    o3d.io.write_point_cloud("pc_12.ply", pcd1+pcd2)
    #print(rgbd_image)

def manual_test_two_ply(color_path1, depth_path1, color_path2, depth_path2,
                        pose_quat1, pose_quat2,  intrinsics):
    
    color1 = TumFormatDataLoader.image_read(color_path1)
    depth1 = TumFormatDataLoader.depth_read(depth_path1)
    color2 = TumFormatDataLoader.image_read(color_path2)
    depth2 = TumFormatDataLoader.depth_read(depth_path2)
    cv2.imwrite("img1.png", color1)
    cv2.imwrite("img2.png", color2)
    
    pose1 = xyzquat_to_pose_mat(pose_quat1)
    pose2 = xyzquat_to_pose_mat(pose_quat2)
    pose_f1_to_f2 = (np.linalg.inv(pose1) @ pose2)
    fx = intrinsics[ 0]
    fy = intrinsics[ 1]
    cx = intrinsics[ 2]
    cy = intrinsics[ 3]
    K = np.array([[fx, 0, cx],[0, fy, cy], [0,0,1]])

    xyz1, color_pc1 = TumFormatDataLoader.rgbd_to_point_cloud(color1, depth1, intrinsics, edge_only=False,
                                                              color_path=color_path1,
                                                              depth_path=depth_path1,
                                                              target_num_pts=1024)
    xyz2, color_pc2 = TumFormatDataLoader.rgbd_to_point_cloud(color2, depth2, intrinsics, edge_only=False,
                                                              color_path=color_path2,
                                                              depth_path=depth_path2,
                                                              target_num_pts=1024)
    
    TumFormatDataLoader.stack_two_pc(xyz1, color_pc1, xyz2, color_pc2, pose_f1_to_f2)


def save_color_ply(xyz2, color_pc2, name):

    if isinstance(xyz2, torch.Tensor):
        xyz2 = xyz2.detach().cpu().numpy()
    if isinstance(color_pc2, torch.Tensor):
        color_pc2 = color_pc2.detach().cpu().numpy()
    
    pcd2 = o3d.geometry.PointCloud  ()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    if color_pc2 is not None:
        pcd2.colors = o3d.utility.Vector3dVector(color_pc2)
    o3d.io.write_point_cloud(name, pcd2)

def merge_two_index():
    opt = gen_options()
    opt.exp_args.dataset_path = "/home/"+  getpass.getuser() + "/data/eth3d/"
    opt.batch_size = 1
    opt.exp_args.odom_covis_thresh = 0.95
    opt.exp_args.edge_only = False
    
    db_train = TumFormatDataLoader('eth3d',
                                   opt.exp_args.dataset_path, 'train',
                                   covis_thresh=opt.exp_args.odom_covis_thresh,
                                   num_point=opt.exp_args.num_point)

    db_train.merge_index("data_loader/scene_cache/train.pickle", "data_loader/index_cache/eth3d/scene_cache_0.95_f2f.pickle" )
    

    
    
if __name__ == "__main__":
    #construct_db_index()
    #filter_out_invalid_pairs(False)
    filter_index(mode='eval_traj', target_mode='eval_traj')
    #construct_traj_index()
    #manual_test_two_ply("/home/rayzhang/data/eth3d/training/cables_3/rgb/11878.818605.png",
    #                    "/home/rayzhang/data/eth3d/training/cables_3/depth/11878.818605.png",
    #                    "/home/rayzhang/data/eth3d/training/cables_3/rgb/11878.855469.png",
    #                    "/home/rayzhang/data/eth3d/training/cables_3/depth/11878.855469.png",
    #                    np.array([0.96408611103069, -1.195827705506, 1.4096046316075, 0.82660417595313, 0.37678661548851, -0.24337032490794 ,-0.33989449486534]),
    #                    np.array([0.91992128781448, -1.195956128163, 1.4068590952653, 0.82546570305187, 0.3775004801065, -0.25360054547037, -0.33434491762486]),
    #                    np.array([726.28741455078, 726.28741455078, 354.6496887207, 186.46566772461]))
    #manual_test_two_ply("/home/rayzhang/data/eth3d/training/sfm_lab_room_1/rgb/1540991549.713384.png",
    #                    "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/depth/1540991549.713384.png",
    #                    "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/rgb/1540991550.376936.png",
    #                    "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/depth/1540991550.376936.png",
    #                    np.array([-0.6621445627378, 0.71068724792878, -1.8123524397334, 0.11059120855547, -0.70743430455013, -0.30689187087284, 0.62699574873207]),
    #                    np.array([-0.7866062582396, 0.75950218495303, -1.8203645875267, 0.061780604100903, -0.61619223172784, -0.30825952372253, 0.72212627465669]),
    #                    np.array([726.21081542969, 726.21081542969, 359.2048034668, 202.47247314453]))
    #open3d_stack_pc_from_rgbd("/home/rayzhang/data/eth3d/training/cables_3/rgb/11878.818605.png",
    #                          "/home/rayzhang/data/eth3d/training/cables_3/depth/11878.818605.png",
    #                          "/home/rayzhang/data/eth3d/training/cables_3/rgb/11878.855469.png",
    #                          "/home/rayzhang/data/eth3d/training/cables_3/depth/11878.855469.png",
    #                          np.array([0.96408611103069, -1.195827705506, 1.4096046316075, 0.82660417595313, 0.37678661548851, -0.24337032490794 ,-0.33989449486534]),
    #                          np.array([0.91992128781448, -1.195956128163, 1.4068590952653, 0.82546570305187, 0.3775004801065, -0.25360054547037, -0.33434491762486]),
    #                          np.array([726.28741455078, 726.28741455078, 354.6496887207, 186.46566772461]))
    
    #open3d_stack_pc_from_rgbd("/home/rayzhang/data/eth3d/training/sfm_lab_room_1/rgb/1540991549.713384.png",
    #                          "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/depth/1540991549.713384.png",
    #                          "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/rgb/1540991550.376936.png",
    #                          "/home/rayzhang/data/eth3d/training/sfm_lab_room_1/depth/1540991550.376936.png",
    #                          np.array([-0.6621445627378, 0.71068724792878, -1.8123524397334, 0.11059120855547, -0.70743430455013, -0.30689187087284, 0.62699574873207]),
    #                          np.array([-0.7866062582396, 0.75950218495303, -1.8203645875267, 0.061780604100903, -0.61619223172784, -0.30825952372253, 0.72212627465669]),
    #                          np.array([726.21081542969, 726.21081542969, 359.2048034668, 202.47247314453]))
    #print_dataindex_len()
    #filter_high_overlap_pairs(0.95, "overlap_pairs_0.95.txt")
    #filter_high_overlap_pairs(0.9, "overlap_pairs_0.9.txt")
    #filter_high_overlap_pairs(0.8, "overlap_pairs_0.8.txt")
    #filter_high_overlap_pairs(0.7, "overlap_pairs_0.7.txt")
    #filter_high_overlap_pairs(0.6, "overlap_pairs_0.6.txt")
    #filter_high_overlap_pairs(0.5, "overlap_pairs_0.5.txt")
    #merge_two_index()
    #balance_dataindex_seq_num()
                
    




