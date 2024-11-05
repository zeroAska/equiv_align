
import time
from model.vnn.vn_dgcnn import VnDgcnn
from model.option import gen_options
from data_loader.modelnet import ModelNetDataLoader
import torch
import pdb, ipdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pypose as pp
import pypose.optim as ppos
import getpass
import numpy as np
import matplotlib
from .feat_pose_plot import Arrow3D, _arrow3D
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
setattr(Axes3D, 'arrow3D', _arrow3D)


def visualize_vector_field(pc1, feat1=None,
                           pc2=None, feat2=None,
                           num_vis_feats=-1, # -1 means only shows the largest norm features
                           num_vis_points=1024,
                           file_prefix="",
                           xyz_limit=0.75):
    fig = plt.figure(figsize=(8, 8))

    #================
    # arrow atributes
    #================
    color1 = "dodgerblue"
    color2 = "r"
    color1_rot = "g"
    arrow_scale = 0.005 #np.linalg.norm(feat1[0,0,:,0]) / 2
    arrow1_attr= dict(mutation_scale=3, lw=1, arrowstyle='-|>', color=color1, linestyle='dotted')
    arrow2_attr= dict(mutation_scale=3, lw=1, arrowstyle='-|>', color=color2, linestyle='dashed')
    axis_min = -2
    axis_max = 2
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 1, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    title_prefix = file_prefix + ": Source and Target:"
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-xyz_limit,xyz_limit)
    ax.set_ylim(-xyz_limit,xyz_limit)
    ax.set_zlim(-xyz_limit,xyz_limit)
    #ax.set_title(title_prefix + ' Features from Equiv Network distance is '+str(np.linalg.norm(feat1-feat2) ))    
    #pdb.set_trace()

    ### pc1
    ### pc1 shape: [ batch,  3, num_pts]
    if pc1 is not None:
        pc1 = pc1[0,:,].transpose()
        X1 = pc1[:num_vis_points, 0]
        Y1 = pc1[:num_vis_points, 1]
        Z1 = pc1[:num_vis_points, 2]
        pc1_viz = ax.scatter(X1, Y1, Z1, marker='o', color=color1, alpha=1)#, label="pc1_features")
        if feat1 is not None:
            dX1 = feat1[0, :num_vis_feats, 0, :num_vis_points]
            dY1 = feat1[0, :num_vis_feats, 1, :num_vis_points]
            dZ1 = feat1[0, :num_vis_feats, 2, :num_vis_points]
            for v in range(num_vis_points):
                for c in range(num_vis_feats):
                    di = np.array([dX1[c, v], dY1[c, v],  dZ1[c, v]])
                    di = di / np.linalg.norm(di) / 4
                    ax.arrow3D(X1[v], Y1[v], Z1[v],
                               di[0], di[1], di[2],
                               **arrow1_attr)
          
    

    #if feat2 is not None and pc2 is not None:
    if pc2 is not None:
        pc2 = pc2[0,:,].transpose()
        X2 = pc2[:num_vis_points, 0]
        Y2 = pc2[:num_vis_points, 1]
        Z2 = pc2[:num_vis_points, 2]
        #pc2_viz = ax.scatter(X2, Y2, Z2, marker='^', color="r", alpha=0.3, label="Target features")
        pc_viz2 = ax.scatter(X2, Y2, Z2, marker='o', color=color2, alpha=1) #label="features, from transformed points inputs through the network" )        
        if feat2 is not None:
            dX2 = feat2[0, :num_vis_feats, 0, :num_vis_points]
            dY2 = feat2[0, :num_vis_feats, 1, :num_vis_points]
            dZ2 = feat2[0, :num_vis_feats, 2, :num_vis_points]

            for v in range(num_vis_points):
                for c in range(num_vis_feats):
                    di = np.array([dX2[c, v], dY2[c, v],  dZ2[c, v]])
                    di = di / np.linalg.norm(di) / 4
                    ax.arrow3D(X2[v],Y2[v], Z2[v],
                               di[0], di[1], di[2],
                               **arrow2_attr)
        
        

    #ax.legend()
    plt.show()
    #time.sleep(0.1)
    #plt.pause(0.0001)
    plt.axis('off')
    plt.savefig(file_prefix + ".png")
    

def visualize_before_after(pc1, feat1, pc2, feat2, feat_rotated_directed):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(15, 12))
    
    #===============
    # PC subplot
    #===============

    title_prefix = "Before rotation:"
    ax = fig.add_subplot(2, 2, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="input point clouds, unrotated")
    #X = pc2[:, 0]
    #Y = pc2[:, 1]
    #Z = pc2[:, 2]
    #pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')

    title_prefix = "After rotation:"
    ax = fig.add_subplot(2, 2, 3, projection=Axes3D.name)#projection='3d')
    ax.view_init(60,60)
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')  
    title_prefix = "Before rotation:"    
    X = feat1[0, 0, :]
    Y = feat1[0, 1, :]
    Z = feat1[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="dodgerblue", label="features, unrotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')

    ax = fig.add_subplot(2, 2, 4, projection='3d')  
    title_prefix = "After rotation:"    
    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')


    X = feat3[0, 0, :]
    Y = feat3[0, 1, :]
    Z = feat3[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="g", alpha=0.3, label="feature map directly beging rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_title(title_prefix + ': Features from Equiv Network')

    ax.legend()

    ipdb.set_trace()
    plt.show()



def visualize_before_after_2(pc1, feat1, pc2, feat2):
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(20, 8))
    
    #===============
    # PC subplot
    #===============
    title_prefix = "Before and After rotation:"
    ax = fig.add_subplot(1, 2, 1, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    #ax.view_init(60,60)
    X = pc1[:, 0]
    Y = pc1[:, 1]
    Z = pc1[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="dodgerblue", label="input point clouds, unrotated")
    X = pc2[:, 0]
    Y = pc2[:, 1]
    Z = pc2[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="r", label="input point clouds, rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    ax.legend()
    
    #===============
    # feature subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection=Axes3D.name)#'3d')
    ax.view_init(60,60)
    title_prefix = "After rotation:"    
    X = feat1[0, 0, :]
    Y = feat1[0, 1, :]
    Z = feat1[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^', color="r", alpha=0.3, label="features, from rotated points inputs through the network" )

    X = feat2[0, 0, :]
    Y = feat2[0, 1, :]
    Z = feat2[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='o', color="g", alpha=0.3, label="feature map directly beging rotated")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network distance is '+str(np.linalg.norm(feat1-feat2) ))    

    ax.legend()
    #plt.show()
    plt.savefig("1.png")


def visualize_before_after_3(pc1, feat1, pc2, feat2, feat3, num_vis_points, num_vis_feats):

    ### pc1 shape: [ batch,  3, num_pts]
    pc1 = pc1[0,:,].transpose()
    pc2 = pc2[0,:,].transpose()
    
    # set up a figure twice as wide as it is tall
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(18, 5))

    #================
    # arrow atributes
    #================
    color1 = "dodgerblue"
    color2 = "r"
    color1_rot = "g"
    arrow_scale = 0.005 #np.linalg.norm(feat1[0,0,:,0]) / 2
    arrow1_attr= dict(mutation_scale=3, lw=1, arrowstyle='-|>', color=color1, linestyle='dotted')
    arrow2_attr= dict(mutation_scale=3, lw=1, arrowstyle='-|>', color=color2, linestyle='dashed')
    axis_min = -2
    axis_max = 2
    
    #===============
    # PC subplot
    #===============
    title_prefix = "Before and after transformation: "
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    #ax.view_init(60,60)
    X1 = pc1[:num_vis_points, 0]
    Y1 = pc1[:num_vis_points, 1]
    Z1 = pc1[:num_vis_points, 2]
    pc_viz = ax.scatter(X1, Y1, Z1, marker='o', color=color1, label="input point clouds, not transformed")
    X2 = pc2[:num_vis_points, 0]
    Y2 = pc2[:num_vis_points, 1]
    Z2 = pc2[:num_vis_points, 2]
    pc_viz = ax.scatter(X2, Y2, Z2, marker='o', color=color2, label="input point clouds, transformed")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(axis_min,axis_max)
    ax.set_ylim(axis_min,axis_max)
    ax.set_zlim(axis_min,axis_max)
    
    ax.set_title(title_prefix + ': Raw 3D point cloud input \n t=[0,0,1], R=[0, 0.717, 0, 0.717] ')
    ax.legend()
     
    
    #===============
    # feature subplot
    # Vnn feat format: [batch, channel, 3, num_points]
    #===============
    ax = fig.add_subplot(1, 3, 2, projection='3d')  
    title_prefix = "Features before and After rotation"    
    dX1 = feat1[0, :num_vis_feats, 0, :num_vis_points]
    dY1 = feat1[0, :num_vis_feats, 1, :num_vis_points]
    dZ1 = feat1[0, :num_vis_feats, 2, :num_vis_points]
    pc_viz = ax.scatter(X1, Y1, Z1, marker='o', color=color1, alpha=0.3, label="features, not transformed")
    for v in range(num_vis_points):
        for c in range(num_vis_feats):
            #pdb.set_trace()
            ax.arrow3D(X1[v], Y1[v], Z1[v],
                       dX1[c, v], dY1[c, v],  dZ1[c, v],
                       **arrow1_attr)
          

    dX2 = feat2[0, :num_vis_feats, 0, :num_vis_points]
    dY2 = feat2[0, :num_vis_feats, 1, :num_vis_points]
    dZ2 = feat2[0, :num_vis_feats, 2, :num_vis_points]
    pc_viz2 = ax.scatter(X2, Y2, Z2, marker='o', color=color2, alpha=0.3, label="features, from transformed points inputs through the network" )
    for v in range(num_vis_points):
        for c in range(num_vis_feats):
            ax.arrow3D(X2[v],Y2[v], Z2[v],
                       dX2[c, v], dY2[c, v],  dZ2[c, v],
                       **arrow2_attr)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(axis_min,axis_max)
    ax.set_ylim(axis_min,axis_max)
    ax.set_zlim(axis_min,axis_max)
    
    
    ax.set_title(title_prefix + ',\n Feature Distance Norm: '+str( np.linalg.norm(feat1-feat2) ))
    ax.legend()


    #===============
    # rotated feature subplot
    # Vnn feat format: [batch, channel, 3, num_points]
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 3, 3, projection='3d')  
    title_prefix = "Features after rotation"
    pc_viz2 = ax.scatter(X2, Y2, Z2, marker='o', color=color2, alpha=0.3, label="features, from transformed points inputs through the network" )
    for v in range(num_vis_points):
        for c in range(num_vis_feats):
            ax.arrow3D(X2[v],Y2[v], Z2[v],
                       dX2[c, v], dY2[c, v],  dZ2[c, v],
                       **arrow2_attr)

    X3 = X2
    Y3 = Y2
    Z3 = Z2
    dX3 = feat3[0, :num_vis_feats, 0, :num_vis_points]
    dY3 = feat3[0, :num_vis_feats, 1, :num_vis_points]
    dZ3 = feat3[0, :num_vis_feats, 2, :num_vis_points]
    pc_viz3 = ax.scatter(X3, Y3, Z3, marker='o', color=color1, alpha=0.3, label="features directed being transformed" )
    for v in range(num_vis_points):
        for c in range(num_vis_feats):
            ax.arrow3D(X3[v],Y3[v], Z3[v],
                       dX3[c, v], dY3[c, v],  dZ3[c, v],
                       **arrow1_attr)
    ax.set_title(title_prefix + ', \nFeature Distance Norm: '+str( np.linalg.norm(feat2-feat3) ))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(axis_min,axis_max)
    ax.set_ylim(axis_min,axis_max)
    ax.set_zlim(axis_min,axis_max)
    
    ax.legend()

    print('plt.show()')
    plt.show()        
    
    

def visualize_pcs(pc : np.ndarray,
                  pc_feat : np.ndarray,
                  title_prefix : str,
                  fig,
                  row_id):

    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = fig.add_subplot(row_id, 2, 1, projection='3d')
    X = pc[:, 0]
    Y = pc[:, 1]
    Z = pc[:, 2]
    pc_viz = ax.scatter(X, Y, Z, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Raw 3D point cloud input')
    #===============
    # Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(row_id, 2, 2, projection='3d')  
  
    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    X = pc_feat[0, 0, :]
    Y = pc_feat[0, 1, :]
    Z = pc_feat[0, 2, :]
    pc_viz = ax.scatter(X, Y, Z, marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title_prefix + ': Features from Equiv Network')



if __name__ == '__main__':
    data = ModelNetDataLoader('/home/'+getpass.getuser()+'/data/modelnet/modelnet40_normal_resampled/',split='test',
                              uniform=False, normal_channel=False,npoint=1024)
    dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    pc = next(iter(dl))
    pc = next(iter(dl)) 
    
    encoder = VnDgcnn(20).cuda()
    checkpoint = torch.load( 'checkpoints/best_model.pth')
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

    ### pc format: [batch, num_points, 3]
    ### Vnn input format: [batch, 3, num_points]
    ### Vnn feat format: [batch, channel, 3, num_points]
    x_in = torch.transpose(pc[0], 2, 1).cuda() # shape: [batch, 3, num_points]

    #pdb.set_trace()
    feat, _ = encoder(x_in)

    pc1 = x_in.cpu().detach().numpy()
    feat1 = feat.cpu().detach().numpy()
    #visualize_pcs(x_in.cpu().detach().numpy(),
    #              feat.cpu().detach().numpy(),
    #              "Before rotation")

    T = pp.SE3([1,0,0, 0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
    R = pp.SO3([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
    
    pc_rotated = R @ pc[0]
    pc_transformed = T @ pc[0]
    x_in = torch.transpose(pc_rotated, 2, 1).cuda()
    feat_rotated, _ = encoder(x_in)
    pc2 = torch.transpose(pc_transformed, 2, 1).cpu().detach().numpy()
    feat2 = feat_rotated.cpu().detach().numpy()

    feat1_rot = torch.transpose(R @ torch.transpose(feat.cpu(), 3, 2), 3, 2).cpu().detach().numpy()
    
    #visualize_pcs(x_in.cpu().detach().numpy(),
    #              feat_rotated.cpu().detach().numpy(),
    #              "After rotation")
    visualize_before_after_3(pc1, feat1, pc2, feat2, feat1_rot, 5, 3)
    #visualize_before_after_2(pc1, feat1, pc2, feat2)
    

    

    

    

    

    

    

    
