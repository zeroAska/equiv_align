import torch
import open3d as o3d
import numpy as np
import pypose as pp
import ipdb

def gicp(source, target, trans_init):
    if isinstance(source, torch.Tensor):
        source = source.detach().squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().squeeze().cpu().numpy()
    if isinstance(trans_init, pp.LieTensor):
        trans_init = trans_init.matrix().squeeze().detach().cpu().numpy()

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(source)
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(target)
    
    reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
        pc1,
        pc2,
        0.2,
        np.squeeze(trans_init),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),        
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))        
        


    ret =  pp.from_matrix(torch.from_numpy(reg_p2p.transformation), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()
    return ret


def icp(source, target, trans_init):
    if isinstance(source, torch.Tensor):
        source = source.detach().squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().squeeze().cpu().numpy()
    if isinstance(trans_init, pp.LieTensor):
        trans_init = trans_init.matrix().squeeze().detach().cpu().numpy()

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(source)
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(target)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1,
        pc2,
        0.2,
        np.squeeze(trans_init),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),        
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))        
        


    ret =  pp.from_matrix(torch.from_numpy(reg_p2p.transformation), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    return ret


def preprocess_fpfh_feature(pcd):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20))
    return pcd_fpfh

def point_to_plane_icp(source, target, trans_init):
    if isinstance(source, torch.Tensor):
        source = source.detach().squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().squeeze().cpu().numpy()
    if isinstance(trans_init, pp.LieTensor):
        trans_init = trans_init.matrix().squeeze().detach().cpu().numpy()

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(source)
    pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc1.normalize_normals()

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(target)
    pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc2.normalize_normals()
    
    result = o3d.pipelines.registration.registration_icp(
        pc1, pc2, 
        1.0,np.squeeze(trans_init),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    try:
        ret = pp.from_matrix(torch.from_numpy(result.transformation), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    except:
        ret = np.eye(4)
        ret[:3,3] = result.transformation[:3,3]
        ret = pp.from_matrix(torch.from_numpy(ret), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    return ret
def fpfh_ransac(source, target, trans_init):
    if isinstance(source, torch.Tensor):
        source = source.detach().squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().squeeze().cpu().numpy()
    if isinstance(trans_init, pp.LieTensor):
        trans_init = trans_init.matrix().squeeze().detach().cpu().numpy()

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(source)
    pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc1.normalize_normals()

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(target)
    pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc2.normalize_normals()
    
    
    source_fpfh = preprocess_fpfh_feature(pc1 )
    target_fpfh = preprocess_fpfh_feature(pc2 )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pc1, pc2, source_fpfh, target_fpfh, True,
        0.2, #2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                1.0)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000,0.9999))#(10000, 0.999))
    try:
        ret = pp.from_matrix(torch.from_numpy(result.transformation), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    except:
        ret = np.eye(4)
        ret[:3,3] = result.transformation[:3,3]
        ret = pp.from_matrix(torch.from_numpy(ret), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    return ret

def fpfh_fgr(source, target, trans_init):
    if isinstance(source, torch.Tensor):
        source = source.detach().squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().squeeze().cpu().numpy()
    if isinstance(trans_init, pp.LieTensor):
        trans_init = trans_init.matrix().squeeze().detach().cpu().numpy()

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(source)
    pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc1.normalize_normals()

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(target)
    pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pc2.normalize_normals()
    
    
    source_fpfh = preprocess_fpfh_feature(pc1 )
    target_fpfh = preprocess_fpfh_feature(pc2 )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc1, pc2, source_fpfh, target_fpfh, 
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=0.05 #0.5
            ))
    try:
        ret = pp.from_matrix(torch.from_numpy(result.transformation), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    except:
        ret = np.eye(4)
        ret[:3,3] = result.transformation[:3,3]
        ret = pp.from_matrix(torch.from_numpy(ret), ltype=pp.SE3_type).unsqueeze(dim=0).Inv()

    return ret

def e2pn():
    pass



BASELINES = {
    "icp": icp,
    "fpfh": fpfh_ransac,
    "gicp": gicp,
    "epn": e2pn,
    "point_to_plane": point_to_plane_icp,
    "fgr": fpfh_fgr
}


def baseline(baseline_type, pc1, pc2, T_init):
    print("Run ", baseline_type)
    method = BASELINES[baseline_type]
    pose = method(pc1, pc2, T_init)
    return pose, None, None, True
    

