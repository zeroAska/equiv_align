import torch
from torch import nn
from model.vnn.utils.vn_dgcnn_util import get_graph_feature, centerize
import open3d as o3d

class Preprocessor(nn.Module):
    def __init__(self,
                 is_centerize : bool,
                 is_normalize : bool):
        super(Preprocessor, self).__init__()
        self.is_centerize = is_centerize
        self.is_normalize = is_normalize
        self.debug_mode = False

    ### input shape: x = [batch, 3, num_pts]
    def forward(self, x, y):
        
        if self.is_centerize:
            x, x_translation = centerize(x)
            y, y_translation = centerize(y)
        else:
            x, x_translation = x, torch.zeros(x.shape[:-1]).cuda().detach()
            y, y_translation = y, torch.zeros(y.shape[:-1]).cuda().detach()

        if self.is_normalize:
            x_max_norm = torch.max(torch.norm(x, dim=1), dim=-1)[0].unsqueeze(1)
            y_max_norm = torch.max(torch.norm(y, dim=1), dim=-1)[0].unsqueeze(1)
            normalize_scale = (torch.max(torch.cat((x_max_norm, y_max_norm), dim=-1), dim=-1)[0]).detach()
            print("normalize scale is ",normalize_scale)
            normalize_scale_x = normalize_scale.unsqueeze(1).unsqueeze(1).expand(x.shape)
            normalize_scale_y = normalize_scale.unsqueeze(1).unsqueeze(1).expand(y.shape)

            if self.debug_mode:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(x[0].cpu().detach().numpy().transpose())
                o3d.io.write_point_cloud("pc1_before_normalize.ply", pcd)

            x, y = x / normalize_scale_x, y / normalize_scale_y
            
            if self.debug_mode:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(x[0].cpu().detach().numpy().transpose())
                o3d.io.write_point_cloud("pc1_after_normalize.ply", pcd)
            

            

        else:
            normalize_scale = torch.ones((x.shape[0],)).detach().cuda()

        return x, y, x_translation, y_translation, normalize_scale

    
        

    
