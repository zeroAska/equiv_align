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
import pypose as pp
import numpy as np
import open3d as o3d
import ipdb

import pickle
import sys, os
from model.utils import create_o3d_pc_from_np
import time
from PIL import Image
import glob
import cv2
from scipy.spatial.transform import Rotation as Rot
from data_loader.tum import open3d_pc_from_rgbd

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst



def save_to_gif(iters,
                file_ext="png",
                pc_prefix='',
                img_prefix='',
                file_prefix=''):
    frames = []
    imgs = glob.glob("*."+file_ext)
    for i in range(iters):
        new_pc = Image.open(pc_prefix+str(i)+"."+file_ext)
        new_color = Image.open(img_prefix+str(i)+"."+file_ext)
        new_frame = get_concat_h_blank(new_color, new_pc)#.save('data/dst/pillow_concat_h_blank.jpg')
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(file_prefix+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100, loop=0)



def visualize_traj_rgbd( poses, seq_name, start_frame, 
                        intrinsics,
                        num_frames_to_show=-1,
                        save_image=False):
    
    #target.transform(flip_transform)
    num_frames = min(num_frames_to_show, poses.shape[0])

    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()
    #vis.add_geometry(source)


    img_dir = "/home/rayzhang/data/eth3d/training/"+seq_name + "/rgb/"
    depth_dir  ="/home/rayzhang/data/eth3d/training/"+seq_name + "/depth/"

    lst = os.listdir(img_dir)
    lst.sort()

    
    pc = open3d_pc_from_rgbd(img_dir + "/" + lst[0],
                             depth_dir + "/" + lst[0],
                             intrinsics,
                             target_num_pts=None)
    vis.add_geometry(pc)
    #vis.get_render_option().load_from_json("ScreenCamera_2023-11-27-05-19-19.json")
    ctr = vis.get_view_control()
    #parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2023-11-27-05-31-08.json")
    #ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.capture_screen_image("0.png")

    #fsfcv2.imshow("current img", img)
    #cv2.waitKey(0)
    for i in range(1, num_frames):
        pc = open3d_pc_from_rgbd(img_dir + "/" + lst[i],
                                 depth_dir + "/" + lst[i],
                                 intrinsics,
                                 target_num_pts=None)
        #ipdb.set_trace()
        pose = poses[i, :].reshape((4,4))
        print("frame {}: new pc #{}, pose\n{}\n".format(i, np.asarray(pc.points).shape[0], pose))


        pc.transform(pose)
        vis.add_geometry(pc,False)
        
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image(str(i) + ".png" )


        #img=cv2.imread(img_dir+lst[i])
        #cv2.imshow("current img", img)
        #cv2.imwrite("img_{}.png".format(i), img)
        #cv2.waitKey(0)
        #time.sleep(0.05) 
    vis.run()
    #save_to_gif(num_frames,
    #            img_prefix='img_',
    #           file_prefix='sfm_lab_room_1')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


if __name__ == '__main__':
    pose_fname = sys.argv[1]
    seq_name = sys.argv[2]
    start_frame = sys.argv[3]
    num_frames_to_show = int(sys.argv[4])
    
    poses_34 = np.genfromtxt(pose_fname)
    poses = np.zeros((poses_34.shape[0], 4, 4))
    for i in range(poses_34.shape[0]):
        #print(poses_34[i, :])
        poses[i, :, :] = np.eye(4)
        poses[i, :3, :] = np.reshape(poses_34[i, :], (3,4))

    intrinsics = np.array([726.28741455078, 726.28741455078, 354.6496887207,  186.46566772461])
        

    visualize_traj_rgbd( poses,
                        seq_name, start_frame,
                        intrinsics,
                        num_frames_to_show=num_frames_to_show,
                        save_image=False)
            
    
