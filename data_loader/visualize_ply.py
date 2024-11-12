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
import open3d as o3d
import numpy as np
import sys, os


if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()

    pcds = []
    for i in range(1, len(sys.argv)):
        pcd = o3d.io.read_point_cloud(sys.argv[i])
        pcds.append(pcd)
        print("loaded pcd ", pcd)
        print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries(pcds)
