import numpy as np
import open3d as o3d
import sys, os

if __name__ == '__main__':
    fname = sys.argv[1]

    pcd = o3d.io.read_point_cloud(fname)
    o3d.visualization.draw_geometries([pcd])

