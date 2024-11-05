'''

    Credit: all the functions of this file are borrowed from the Vision-Graphics deep learning ToolKit,
    author: Shichen Liu*, Haiwei Chen*,
    author_email: liushichen95@gmail.com,
    license: MIT License

'''
import os
import trimesh
import random
import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as sciR
import open3d as o3d



'''
Point cloud augmentation
Only numpy function is included for now
'''


def crop_2d_array(pc_in, crop_ratio):

    ind_src = np.random.randint(pc_in.shape[1])
    selected_col = pc_in[:, ind_src]
    ind_order = np.argsort(selected_col)
    sorted_pc = pc_in[ind_order, :]

    head_or_tail = np.random.randint(2)
    
    if head_or_tail:
        crop_ratio = max(crop_ratio, 1-crop_ratio)
        sorted_pc = np.split(sorted_pc, [int(crop_ratio*sorted_pc.shape[0])], axis=0)[0]
    else:
        crop_ratio = min(crop_ratio, 1-crop_ratio)
        sorted_pc = np.split(sorted_pc, [int(crop_ratio*sorted_pc.shape[0])], axis=0)[1]

    return sorted_pc
    


def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))


def translate_point_cloud(data: np.array, max_translation_norm: float):
    """
    Input: Nx3 array
    
    """
    
    T = np.random.rand(1,3) * max_translation_norm

    return data+T, T.transpose().squeeze()

def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R: 
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation 
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)

    if R is not None:
      rotation_angle = R
    elif max_degree is not None and abs(max_degree) > 1e-1:
      #if (max_degree == 0):
      #  rotation_angle = np.zeros_like(data.shape)
      #else:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]
    
    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(rotation_matrix, data.reshape((-1, 3)).T)

    return rotated_data.T, rotation_matrix



def label_relative_rotation_simple(anchors, T, rot_ref_tgt=False):
    """Find the anchor rotation that is closest to the queried rotation. 
    return: 
    R_target: [3,3], T = R_target * anchors[label]
    label: int"""
    if rot_ref_tgt:
        # anchors.T * T = R_target -> T = anchors * R_target
        T_then_anchors = np.einsum('aji,jk->aik', anchors, T)
    else:
        # T * anchors.T = R_target -> T = R_target * anchors
        T_then_anchors = np.einsum('ij,akj->aik', T, anchors)
    label = np.argmax(np.einsum('aii->a', T_then_anchors),axis=0)
    R_target = T_then_anchors[label.item()]
    # return: [3,3], int
    return R_target, label


def label_relative_rotation_np(anchors, T):
    """For all anchor rotations, it finds their corresponding anchor rotation such that the difference between two rotations is closest to the queried rotation.
    They are used as the rotation classification label. 
    return: 
    R_target: [60,3,3]
    label: [60]"""
    T_from_anchors = np.einsum('abc,bj,ijk -> aick', anchors, T, anchors)
    # R_res = Ra^T R Ra (Ra R_res = R Ra)
    label = np.argmax(np.einsum('abii->ab', T_from_anchors),axis=1)
    idxs = np.vstack([np.arange(label.shape[0]), label]).T
    R_target = T_from_anchors[idxs[:,0], idxs[:,1]]
    return R_target, label


'''
    (B)x3x3, Nx3x3 -> dist, idx
'''
def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        # diff_r = np.matmul(r0, r1.T)
        # return np.einsum('ii', diff_r)

        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)

        return traces, np.argmax(traces), diff_r



'''
Point cloud transform:
    pc: 
        torch: [b, 3, p]
        np: [(b, )3, p]
'''

# translation normalization
def centralize(pc):
    return pc - pc.mean(dim=2, keepdim=True)

def centralize_np(pc, batch=False):
    axis = 2 if batch else 1
    return pc - pc.mean(axis=axis, keepdims=True)


def normalize(pc):
    """Centralize and normalize to a unit ball. Take a batched pytorch tensor. """
    pc = centralize(pc)
    var = pc.pow(2).sum(dim=1, keepdim=True).sqrt()
    return pc / var.max(dim=2, keepdim=True)

def normalize_np(pc, batch=False):
    """Centralize and normalize to a unit ball. Take a numpy array. """
    pc = centralize_np(pc, batch)
    axis = 1 if batch else 0
    var = np.sqrt((pc**2).sum(axis=axis, keepdims=True))
    return pc / var.max(axis=axis+1, keepdims=True)

def uniform_resample_index_np(pc, n_sample, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    n_point = pc.shape[0]
    if n_point >= n_sample:
        # downsample
        idx = np.random.choice(n_point, n_sample, replace=False)
    else:
        # upsample
        idx = np.random.choice(n_point, n_sample-n_point, replace=True)
        idx = np.concatenate((np.arange(n_point), idx), axis=0)
    return idx

def uniform_resample_np(pc, n_sample, label=None, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    idx = uniform_resample_index_np(pc, n_sample, batch)
    if label is None:
        return idx, pc[idx]
    else:
        return idx, pc[idx], label[idx]



def get_so3_from_anchors_np(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    sbeta = face_normals[...,-1]
    cbeta = (1 - sbeta**2)**0.5
    calpha = face_normals[...,0] / cbeta
    salpha = face_normals[...,1] / cbeta

    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = -gamma[None].repeat(na, axis=0)

    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)

    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = salpha
    Rz[:,2] = 0
    Rz[:,3] = -salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1

    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta

    # B Matrix
    Rx[:,:,0] = 1
    Rx[:,:,1] = 0
    Rx[:,:,2] = 0
    Rx[:,:,3] = 0
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = np.sin(gamma)
    Rx[:,:,6] = 0
    Rx[:,:,7] = -np.sin(gamma)
    Rx[:,:,8] = np.cos(gamma)

    padding = 60
    Rx2[:,:,0] = 1
    Rx2[:,:,1] = 0
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = 0
    Rx2[:,:,4] = np.cos(gamma+padding/180*np.pi)
    Rx2[:,:,5] = np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = -np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,8] = np.cos(gamma+padding/180*np.pi)

    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)

    # R = BCD
    Rxy = np.einsum('bij,bjh->bih', Rx, Ry)
    Rxy2 = np.einsum('bij,bjh->bih', Rx2, Ry)
    Rs1 = np.einsum('bij,bjh->bih', Rxy, Rz)
    Rs2 = np.einsum('bij,bjh->bih', Rxy2, Rz)

    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    # import ipdb; ipdb.set_trace()
    Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
         Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
    return Rs


def get_so3_from_anchors_np_zyz(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    cbeta = face_normals[...,-1]
    sbeta = (1 - cbeta**2)**0.5
    calpha = face_normals[...,0] / sbeta
    salpha = face_normals[...,1] / sbeta

    if gsize==5:
        calpha = np.where(np.isnan(calpha) & (cbeta>0), np.ones_like(calpha), calpha)
        calpha = np.where(np.isnan(calpha) & (cbeta<0), -np.ones_like(calpha), calpha)
        salpha = np.where(np.isnan(salpha), np.zeros_like(salpha), salpha)

    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = gamma[None].repeat(na, axis=0)

    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)
    # Rx3 = np.zeros([na, gsize, 9], dtype=np.float32)
    # Rx4 = np.zeros([na, gsize, 9], dtype=np.float32)

    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = -salpha
    Rz[:,2] = 0
    Rz[:,3] = salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1

    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta

    # B Matrix
    Rx[:,:,0] = np.cos(gamma)
    Rx[:,:,1] = -np.sin(gamma)
    Rx[:,:,2] = 0
    Rx[:,:,3] = np.sin(gamma)
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = 0
    Rx[:,:,6] = 0
    Rx[:,:,7] = 0
    Rx[:,:,8] = 1

    # padding = 60  # hardcoded for gsize=3
    padding = 2 * np.pi / gsize / 2 # adaptive to gsize
    Rx2[:,:,0] = np.cos(gamma+padding) #/180*np.pi
    Rx2[:,:,1] = -np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,4] = np.cos(gamma+padding) #/180*np.pi
    Rx2[:,:,5] = 0
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = 0
    Rx2[:,:,8] = 1

    # Rx3[:,:,0] = np.cos(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,1] = -np.sin(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,2] = 0
    # Rx3[:,:,3] = np.sin(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,4] = np.cos(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,5] = 0
    # Rx3[:,:,6] = 0
    # Rx3[:,:,7] = 0
    # Rx3[:,:,8] = 1

    # Rx4[:,:,0] = np.cos(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,1] = -np.sin(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,2] = 0
    # Rx4[:,:,3] = np.sin(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,4] = np.cos(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,5] = 0
    # Rx4[:,:,6] = 0
    # Rx4[:,:,7] = 0
    # Rx4[:,:,8] = 1

    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)
    # Rx3 = Rx3.reshape(na*gsize,3,3)
    # Rx4 = Rx4.reshape(na*gsize,3,3)

    Ryx = np.einsum('bij,bjh->bih', Ry, Rx)
    Ryx2 = np.einsum('bij,bjh->bih', Ry, Rx2)
    # Ryx3 = np.einsum('bij,bjh->bih', Ry, Rx3)
    # Ryx4 = np.einsum('bij,bjh->bih', Ry, Rx4)
    Rs1 = np.einsum('bij,bjh->bih', Rz, Ryx)
    Rs2 = np.einsum('bij,bjh->bih', Rz, Ryx2)
    # Rs3 = np.einsum('bij,bjh->bih', Rz, Ryx3)
    # Rs4 = np.einsum('bij,bjh->bih', Rz, Ryx4)

    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    # import ipdb; ipdb.set_trace()
    if gsize == 3:
        Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
            Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
        # -0.7947, -0.1876, 0.1876, 0.7967
        # each will make only one of the four conditions true
    elif gsize == 5:
        Rs = Rs2*(np.abs(z_val+1)<0.01)+Rs1*(np.abs(z_val+0.447)<0.01)+\
            Rs2*(np.abs(z_val-0.447)<0.01)+Rs1*(np.abs(z_val-1)<0.01)
        # Rs = Rs1
    else:
        raise NotImplementedError('gsizee other than 3 (for faces) or 5 (for vertices) are not supported: %d'%gsize)
    return Rs

# functions for so3 sampling
def get_adjmatrix_trimesh(mesh, gsize=None):
    face_idx = mesh.faces
    face_adj = mesh.face_adjacency
    adj_idx = []
    binary_swap = np.vectorize(lambda a: 1 if a == 0 else 0)
    for i, fidx in enumerate(face_idx):
        fid = np.argwhere(face_adj == i)
        fid[:,1] = binary_swap(fid[:,1])
        adj_idx.append(face_adj[tuple(np.split(fid, 2, axis=1))].T)

    face_adj =  np.vstack(adj_idx).astype(np.int32)

    if gsize is None:
        return face_adj
    else:
        # Padding with in-plane rotation neighbors
        na = face_adj.shape[0]
        R_adj = (face_adj * gsize)[:,None].repeat(gsize, axis=1).reshape(-1,3)
        R_adj = np.tile(R_adj,[1,gsize]) + np.arange(gsize).repeat(3)[None].repeat(na*gsize, axis=0)
        rp = (np.arange(na) * gsize).repeat(gsize)[..., None].repeat(gsize,axis=1)
        rp = rp + np.arange(gsize)[None].repeat(na*gsize,axis=0)
        R_adj = np.concatenate([R_adj, rp], axis=1)
        return R_adj


def get_adjmatrix_trimesh_vtx(mesh, gsize=None):
    """return: nparray of size (12,5) (12,5) (12,)"""
    vertices = mesh.vertices    # 12,3
    vtx_adj = mesh.edges    # 60*2 (each edge has the reverse pair as well)
    v_neighbors = mesh.vertex_neighbors # 12,5 ndarray

    level_2s = []
    opposites = []
    for i in range(vertices.shape[0]):
        counted = []
        counted.append(i)
        vns = v_neighbors[i]
        counted.extend(vns)
        level_2 = []
        for vn in vns:
            vnns = v_neighbors[vn]
            for vnn in vnns:
                if vnn not in counted:
                    level_2.append(vnn)
                    counted.append(vnn)
        level_2s.append(level_2)
        for j in range(vertices.shape[0]):
            if j not in counted:
                opposites.append(j)
                break

    level_2s = np.array(level_2s)   # 12,5
    opposites = np.array(opposites) # 12
    return v_neighbors, level_2s, opposites

def icosahedron_so3_trimesh(mesh_path, gsize=3, use_quats=False):
    # 20 faces, 12 vertices
    # root = vgtk.__path__[0]
    # mesh_path = os.path.join(root, 'data', 'anchors/sphere12.ply')
    mesh = trimesh.load(mesh_path)
    mesh.fix_normals()
    face_idx = mesh.faces
    face_normals = mesh.face_normals

    fix_angle = np.arctan(face_normals[9, 2] / face_normals[9, 0])
    fix_rot = np.array([[np.cos(fix_angle),  0,  np.sin(fix_angle)],
                        [0,                  1,  0],
                        [-np.sin(fix_angle), 0, np.cos(fix_angle)]])
    # face_normals = face_normals @ fix_rot.T

    na = face_normals.shape[0]
    # gsize = 3

    # 60x3x3
    Rs = get_so3_from_anchors_np(face_normals, gsize=gsize) # .reshape(na, gsize, 3, 3)
    # 60x12
    Rs = np.einsum('bij,kj', Rs, Rs[29])
    R_adj = get_adjmatrix_trimesh(mesh, gsize)

    # 60x12x3x3
    grouped_R = np.take(Rs, R_adj, axis=0)

    # relative_Rs = np.einsum('bkij,bjh->bkih', grouped_R, np.transpose(Rs,(0,2,1)))

    # # 12x3x3
    # canonical_R = relative_Rs[0]
    # nn = canonical_R.shape[0]

    # # 60x12x3x3
    # ordered_R = np.einsum('kij,bkjh->bkih',canonical_R, Rs[:,None].repeat(nn, axis=1))


    ################

    relative_Rs = np.einsum('kjh,lh->kjl', grouped_R[0], Rs[0]) # 12x3x3
    # relative_Is = np.einsum('', relative_Rs, )
    ordered_R = np.einsum('kmj,bji->bkim', relative_Rs, Rs) # 60x12x3x3
    # ordered_R = np.einsum('kmj,bij,kli->bkml', relative_Rs, Rs, relative_Rs) # 60x12x3x3
    # ordered_R = np.einsum('bml,kmj,bij->bkli', Rs, relative_Rs, Rs)
    # ordered_R = np.einsum('bkmi,kjm,bkjl->bkli', ordered_R, relative_Rs, ordered_R)
    canonical_R = None

    #################

    # grouped_R = np.einsum('kij,bkjh->bkih', relative_Rs, grouped_R)

    # 60x12x1x3x3, 60x1x12x3x3 -> 60x12x12x3x3 -> 60x12x1 argmin diff
    tiled_ordr = np.expand_dims(ordered_R,axis=2)

    ###
    diff_r = np.einsum('bkgij,chj->bkcih', tiled_ordr, Rs)

    ## stop using grouped_R
    # tiled_grpr = np.expand_dims(grouped_R,axis=1)
    # # 60x12x12x3x3
    # diff_r = np.einsum('bkgij,bkghj->bkgih', tiled_ordr, tiled_grpr)
    ## stop end

    trace = 0.5 * (np.einsum('bkgii->bkg', diff_r) - 1)
    # 60x12 true index wrt ordered_R
    trace_idx = np.argmax(trace,axis=2)


    # import ipdb; ipdb.set_trace()
    reverse_Rs_idx = np.argmax(np.einsum('nij,mjk->nmji', Rs, Rs).sum(2).sum(2), axis=1)
    trace_idx = trace_idx[reverse_Rs_idx]


    use_idx = [2,3,6,9]
    new_trace_idx = np.zeros([trace_idx.shape[0], len(use_idx)], dtype=np.int32)

    for i in range(trace_idx.shape[0]):
        # trace_idx[i] = R_adj[i][trace_idx[i]]
        new_trace_idx[i] = trace_idx[i,use_idx]

    # ---------------- DEBUG ONLY -------------------------
    # np.set_printoptions(precision=2, suppress=True)
    # print(sciR.from_matrix(np.matmul(ordered_R[1], Rs[1].T)).as_quat())
    # for i in range(30):
    #     print(sciR.from_matrix(np.matmul(Rs[trace_idx[i]], Rs[i].T)).as_quat())
    # import ipdb; ipdb.set_trace()
    # -----------------------------------------------------
    # trace_idx = np.arange(60)[:,None].astype(np.int32)

    if use_quats:
        Rs = sciR.from_matrix(Rs).as_quat()
    # Rs = np.transpose(Rs, [0,2,1])

    reverse_trace_idx = np.zeros_like(new_trace_idx)
    for i in range(new_trace_idx.shape[1]):
        for j in range(new_trace_idx.shape[0]):
            reverse_trace_idx[new_trace_idx[j,i], i] = j

    #################### DEBUG ###########################
    # for i in range(100):
    #     randR = sciR.random().as_matrix()
    #     traces, nns = rotation_distance_np(randR, Rs)
    #     nnidx = np.argsort(-traces)[:5]
    #     print(traces[nnidx])
    # import ipdb; ipdb.set_trace()
    ####################################################

    return Rs, trace_idx, canonical_R
    # return Rs, trace_idx, canonical_R
    

def icosahedron_trimesh_to_vertices(mesh_path):
    mesh = trimesh.load(mesh_path)  # trimesh 3.9 does not work. need 3.2
    mesh.fix_normals()
    vs = mesh.vertices  # each vertex is of norm 1
    vs = np.array(vs)

    # the 5 rotation matrices for each of the 12 vertices
    Rs = get_so3_from_anchors_np_zyz(vs, gsize=5)
    # Rs = Rs.reshape(vs.shape[0], 5, 3, 3)
    # the index of the opposite vertex and the two five-vertex-ring for each vertex
    v_adjs, v_level2s, v_opps = get_adjmatrix_trimesh_vtx(mesh) # 12*5, 12*5, 12
    return vs, v_adjs, v_level2s, v_opps, Rs


GAMMA_SIZE = 3
ANCHOR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'anchors_data', 'sphere12.ply')
vs, v_adjs, v_level2s, v_opps, vRs = icosahedron_trimesh_to_vertices(ANCHOR_PATH)    # 12*3, each vertex is of norm 1
Rs, R_idx, canonical_relative = icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)

def get_anchorsV():
    """return 60*3*3 matrix as rotation anchors determined by the symmetry of icosahedron vertices"""
    return vRs.copy()

def select_anchor(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors

def get_anchors(k=60):
    return select_anchor(Rs,k)

    

def get_relativeV_index() :
    #Rs, vs):
    # the permutation of the 12 vertices under the 60 rotations
    # Rs: 60*3*3, vs: 12*3
    print("Rs.shape", Rs.shape, vs.shape)
    incr_r = np.einsum('dij,aj->dai', Rs, vs) # drotation*anchor, 60*12*3
    incr_r = incr_r[:,:,None]   # 60*12*1*3
    ori_vs = vs[None,None]      # 1*1*12*3

    diff_r = incr_r - ori_vs  # 60*12(rot)*12(ori)*3
    trace = (diff_r**2).sum(-1)  # 60*12*12
    trace_idx_ori = np.argmin(trace,axis=2) # find correspinding original element for each rotated
    trace_idx_rot = np.argmin(trace,axis=1) # find corresponding rotated element for each original

    return trace_idx_ori, trace_idx_rot
