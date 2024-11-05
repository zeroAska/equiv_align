
from PIL import Image
import glob
import sys, os
from model.vnn.vn_dgcnn import VnDgcnn
from model.option import gen_options
from model.equiv_registration import EquivRegistration
from data_loader.modelnet import ModelNetDataLoader
import torch
import pdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pypose as pp
import pypose.optim as ppos

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
#%matplotlib
import matplotlib.pyplot as plt

from model.plot.feat_pose_plot import save_to_gif, visualize_before_after_2





if __name__ == '__main__':
    ####################
    ##  params
    num_files = 5
    iters = 100
    is_using_untrained = True
    ####################
    
    opt = gen_options()
    data = ModelNetDataLoader('data/modelnet/EvenAlignedModelNet40PC/',split='test', uniform=False, normal_channel=False,npoint=1024)
    dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    opt.net_args.num_optimization_iters = 1
    opt.exp_args.batch_size = 1
    dl = iter(dl)

    encoder = EquivRegistration(opt).cuda()    
    ckpt_path = os.path.join(opt.exp_args.pretrained_model_dir, 'best_model.pth')
    checkpoint = torch.load(ckpt_path)
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    

    for file_index in range(num_files):
        pc = next(dl)
        #pc = next(iter(dl))

        T = pp.SO3([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
        T_init = pp.identity_SO3(1).cuda()
        T_init.requires_grad = True
        pc2 = T @ pc[0]

        #convert to [ batch, 3, num_pts]
        pc1_in = torch.transpose(pc[0], 2, 1).cuda()
        pc2_in = torch.transpose(pc2, 2, 1).cuda()
        pc1_in.requires_grad = False
        pc2_in.requires_grad = False

        pc1_np = pc1_in.cpu().detach().numpy()    




        for i in range(iters):

            ### pc format from data loader: [batch, num_points, 3]
            ### Vnn input format: [batch, 3, num_points]
            ### Vnn feat format: [batch, channel, 3, num_points]
            pred, feat1, feat2 = encoder(pc1_in, pc2_in, T_init)


            feat1 = feat1.cpu().detach().numpy() # shape [(batch, channel, 3, num_pt) ]
            #feat2 = feat2.cpu().detach().numpy()
            pc2_np = (T_init[0] @ pc2_in.permute(0,2,1)).permute(0,2,1).cpu().detach().numpy()
            feat2 = torch.transpose(T_init[0] @ torch.transpose(feat2, 3, 2), 3, 2).cpu().detach().numpy() 
        
            visualize_before_after_2(pc1_np, feat1 , pc2_np, feat2 ,  'iter_'+str(i), is_using_untrained)

        save_to_gif(iters,'iter_', str(file_index))

    

    

    

    

    

    

    

    
