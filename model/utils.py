import torch
import torch.nn as nn
import pypose as pp
import open3d as o3d
import numpy as np
import ipdb
def create_o3d_pc_from_np(xyz, color):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.squeeze().detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if color is not None:
        #color_pc2 = np.zeros_like(xyz2)+np.expand_dims(manual_assign_color, axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd
    

def save_two_pc_transformed(xyz1, color_pc1, xyz2, color_pc2,  pose_f1_to_f2,  name_prefix="", is_auto_assign_color=False):
    if  isinstance(pose_f1_to_f2, torch.Tensor) and  torch.numel(pose_f1_to_f2.squeeze()) == 9:
        rot =  pose_f1_to_f2.squeeze().detach().cpu().numpy()
        pose_f1_to_f2 = np.eye(4)
        pose_f1_to_f2[:3, :3] = rot
        

    cvt_to_np = lambda data : data.squeeze().detach().cpu().numpy() if  isinstance(data, torch.Tensor) else data

    xyz1 = cvt_to_np(xyz1)
    xyz2 = cvt_to_np(xyz2)
    pose_f1_to_f2 = cvt_to_np(pose_f1_to_f2)
    if color_pc1 is not None and color_pc2 is not None:
        color_pc1 = cvt_to_np(color_pc1)
        color_pc2 = cvt_to_np(color_pc2)
    if is_auto_assign_color:
        color_pc1 = np.zeros_like(xyz1)+np.expand_dims(np.array([1.0,0.0,0.0]), axis=0)
        color_pc2 = np.zeros_like(xyz2)+np.expand_dims(np.array([0.0,0.0,1.0]), axis=0)

    if xyz1 is not None and xyz2 is not None and pose_f1_to_f2 is not None:

        xyz2 = (pose_f1_to_f2[:3,:3] @ xyz2.transpose() +\
                np.broadcast_to(np.expand_dims(  pose_f1_to_f2[:3, 3], axis=-1) , (3,xyz2.shape[0]))).transpose()
        xyz_full = np.concatenate((xyz1, xyz2), axis=0)
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(xyz_full)        
        if color_pc1 is not None and color_pc2 is not None:        
            color = np.concatenate((color_pc1, color_pc2), axis=0)
            pcd_trans.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(name_prefix+"pc_12.ply", pcd_trans)
        print("write to pc_12.ply")


def save_color_ply(xyz2, color_pc2, name, vector_field_feat=None,
                   manual_assign_color=None
                   ):

    if isinstance(xyz2, torch.Tensor):
        xyz2 = xyz2.squeeze().detach().cpu().numpy()
    
    pcd2 = o3d.geometry.PointCloud  ()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    
    if manual_assign_color is not None and manual_assign_color.size == 3:
        color_pc2 = np.zeros_like(xyz2)+np.expand_dims(manual_assign_color, axis=0)
        pcd2.colors = o3d.utility.Vector3dVector(color_pc2)
        
    elif color_pc2 is not None:
        if isinstance(color_pc2, torch.Tensor):
            color_pc2 = color_pc2.squeeze().detach().cpu().numpy()
        pcd2.colors = o3d.utility.Vector3dVector(color_pc2)

    o3d.io.write_point_cloud(name, pcd2)


def copy_new_leaf_tensor(T_init_in, ltype):
    T_init = T_init_in.detach().clone()
    T_init = torch.empty_like(T_init).copy_(T_init)#T_init_in.requires_grad)
    T_init = pp.LieTensor(T_init.detach(), ltype=ltype).requires_grad_(True)
    return T_init

def filter_ckpt_state_dict(checkpoint_dict,
                           model_dict,
                           filtered_name):
    pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
    for name in filtered_name:
        if (name in model_dict):
            pretrained_dict.pop(name)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    return model_dict

