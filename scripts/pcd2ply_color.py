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
import numpy as np
import open3d as o3d
import sys, os

if __name__ == '__main__':
    fname = sys.argv[1]

    pcd = o3d.io.read_point_cloud(fname)
    o3d.visualization.draw_geometries([pcd])

