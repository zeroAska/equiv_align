
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
from data_loader.rgbd_utils import *


class RegistrationDataset(data.Dataset):
    def __init__(self, name,
                 dataset_path='',
                 run_mode='train',
                 is_user_specified_trainvaltest=False):
        """ Base class for RGBD dataset """
        self.name = name
        self.root = dataset_path
        self.set_run_mode(run_mode)
        
    def set_run_mode(self, mode : str):
        
        #mode_to_dir = {'train': 'training',
        #               'val': 'training',
        #               'test': 'training'}
        #assert mode in mode_to_dir
        self.run_mode = mode
        

        self.seq_path_all = osp.join(self.root, 'training')

        #cur_path = osp.dirname(osp.abspath(__file__))        
        #scene_cache_path = osp.join(cur_path, 'scene_cache', '{}.pickle'.format(self.run_mode))
        #dataindex_cache_path = osp.join(cur_path, 'dataindex_cache', '{}.pickle'.format(self.run_mode))
        

        # building dataset is expensive, cache so only needs to be performed once

    def save_index(self, path_prefix):
        ### get paths for the two index
        #cur_path = osp.dirname(osp.abspath(__file__))        
        scene_cache_path = (path_prefix+ 'scene_cache.pickle')
        dataindex_cache_path = path_prefix + 'dataindex_cache.pickle'

        ### load
        with open(scene_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(scene_cache_path))                
            pickle.dump((self.scene_info,), cachefile)

        with open(dataindex_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(dataindex_cache_path))                
            pickle.dump((self.dataset_index,), cachefile)


    def _load_index(self, scene_cache_path, dataindex_cache_path):

        ### load
        if osp.isfile(scene_cache_path):
            scene_info = pickle.load(open(scene_cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(scene_cache_path, 'wb') as cachefile:
                print("write pickle to {}".format(scene_cache_path))                
                pickle.dump((scene_info,), cachefile)
        self.scene_info = scene_info

        if osp.isfile(dataindex_cache_path):
            self.dataset_index = pickle.load(open(dataindex_cache_path, 'rb'))[0]
        else:
            self.dataset_index = self._build_dataset_index(dataindex_cache_path)

    def merge_index(self, scene_path1, scene_path2):
        ### get paths for the two index
        cur_path = osp.dirname(osp.abspath(__file__))        
        scene_cache_path = osp.join(cur_path, 'scene_cache', '{}.pickle'.format(self.run_mode))
        dataindex_cache_path = osp.join(cur_path, 'dataindex_cache', '{}.pickle'.format(self.run_mode))
        #cur_path = osp.dirname(osp.abspath(__file__))        
        #if not os.path.isdir(osp.join(cur_path, 'scene_cache')):
        #    os.mkdir(osp.join(cur_path, 'scene_cache'))
        #if not os.path.isdir(osp.join(cur_path, 'dataindex_cache')):
        #    os.mkdir(osp.join(cur_path, 'dataindex_cache'))

        ### load
        scene_info1 = pickle.load(open(scene_path1, 'rb'))[0]
        len1 = len(scene_info1)
        scene_info2 = pickle.load(open(scene_path2, 'rb'))[0]
        len2 = len(scene_info2)

        self.scene_info = {**scene_info1, **scene_info2}
        assert len1 + len2 == len(self.scene_info)

        with open(scene_cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(scene_cache_path))                
            pickle.dump((self.scene_info,), cachefile)


        self.dataset_index = self._build_dataset_index(dataindex_cache_path)
            
        
        
                
    def _build_dataset_index(self,cache_path, max_pairs_per_seq=-1):
        dataset_index = {}
        for scene in self.scene_info.keys():
            #dataset_index[scene] = []
            graph = self.scene_info[scene]['frame_pairs']
            if os.path.basename(os.path.normpath(scene)) not in dataset_index:
                dataset_index[os.path.basename(os.path.normpath(scene))] = []

            if max_pairs_per_seq > 0 and len(graph) > max_pairs_per_seq:

                selected_inds = np.random.choice(len(graph), max_pairs_per_seq, replace=False)
                for i in range(selected_inds.size):
                    #dataset_index.append((scene, graph[i]))
                    dataset_index[os.path.basename(os.path.normpath(scene))].append((scene, graph[i]))
            else:
                for i in graph:
                    dataset_index[os.path.basename(os.path.normpath(scene))].append((scene, i))
            #else:
            #    print("Reserving {} for validation".format(scene))
        with open(cache_path, 'wb') as cachefile:
            print("write pickle to {}".format(cache_path))
            pickle.dump((dataset_index,), cachefile)
        return dataset_index

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        if depth_file.endswith(".npy"):
            return np.load(depth_file)
        elif depth_file.endswith(".png"):
            return cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH )

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __len__(self):
        #return len(self.dataset_index)
        if self.access_seq == 'random':
            return sum([len(self.dataset_index[seq_key])for seq_key in self.dataset_index])
        else:
            return len(self.dataset_index[self.access_seq])

    #def __imul__(self, x):
    #    self.dataset_index *= x
    #    return self
    
