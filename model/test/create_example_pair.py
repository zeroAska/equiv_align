import torch
import pypose as pp
import numpy as np
from data_loader.factory import create_datastream
from model.utils import save_two_pc_transformed, save_color_ply
import open3d as o3d
import pickle
from model.option import gen_options
from model.example_opt import create_opt
import getpass

def load_data_pair(filename):
    data = torch.load(filename)
    dataset_name = data['dataset_name']
    pc1 = data['pc1'] 
    pc2 = data['pc2']
    id1 = data['pc1_id'] if 'pc1_id' in data else None
    id2 = data['pc2_id'] if 'pc2_id' in data else None
    color_pc1 = data['color_pc1'] if 'color_pc1' in data else None
    color_pc2 = data['color_pc2'] if 'color_pc2' in data else None
    T_gt = data['T_gt']
    return {
        'dataset_name': dataset_name,
        'pc1': pc1,
        'pc2': pc2,
        'pc1_id': id1,
        'pc2_id': id2,
        'color1': color_pc1,
        'color2': color_pc2,
        'T': T_gt
    }
    
    

def sample_data_pair(dataset_name: str,
                     add_noise : bool,
                     num_pairs : int):

    opt = create_opt(dataset_name, add_noise)
    _, _, testDataLoader = create_datastream(opt)

    for file_index, data in enumerate(testDataLoader):
        pc1 = data['pc1'][:, :, :]
        pc2 = data['pc2'][:, :, :]
        T_gt = data['T']


        color_pc1 = data['color1']         if 'color1' in data else None
        
        color_pc2 = data['color2']         if 'color2' in data else None


        id1 = data['pc1_id'] if 'pc1_id' in data else None
        id2 = data['pc2_id'] if 'pc2_id' in data else None

        save_two_pc_transformed(pc1, color_pc1, pc2, color_pc2,  T_gt,  name_prefix=dataset_name+str(file_index))
        save_two_pc_transformed(pc1, color_pc1, pc2, color_pc2,  np.eye(4),  name_prefix=dataset_name+str(file_index)+"_init")
        
        #save_color_ply(pc1, color_pc1, dataset_name+str(file_index)+"pc1.ply",
        #               manual_assign_color=np.array([1.0,0.0,0.0]))

        #save_color_ply(pc2, color_pc2, dataset_name+str(file_index)+"pc2.ply",
        #               manual_assign_color=np.array([0,0.0,1.0]))
        #import ipdb; ipdb.set_trace()
        torch.save({
            'dataset_name': dataset_name,
            'pc1': pc1,
            'pc2': pc2,
            'pc1_id': id1,
            'pc2_id': id2,
            'color_pc1': color_pc1,
            'color_pc2': color_pc2,
            'T_gt': T_gt
        },dataset_name+"_"+str(file_index)+".pth")
                     



    
if __name__ == '__main__':
    #sample_data_pair('ModelNet40Alignment', True, 1)
    sample_data_pair('eth3d', False, 1)
